use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::{AllQuery, BooleanQuery, Occur, Query, QueryParser, RangeQuery, TermQuery};
use tantivy::schema::{IndexRecordOption, Term, Value};
use tantivy::{Index, IndexReader, TantivyDocument};

use rusqlite::Connection;

use crate::search::tantivy::fields_from_schema;

#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    pub agents: HashSet<String>,
    pub workspaces: HashSet<String>,
    pub created_from: Option<i64>,
    pub created_to: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub title: String,
    pub snippet: String,
    pub score: f32,
    pub source_path: String,
    pub agent: String,
    pub workspace: String,
}

pub struct SearchClient {
    reader: Option<(IndexReader, crate::search::tantivy::Fields)>,
    sqlite: Option<Connection>,
}

impl SearchClient {
    pub fn open(index_path: &Path, db_path: Option<&Path>) -> Result<Option<Self>> {
        let tantivy = Index::open_in_dir(index_path).ok().and_then(|idx| {
            let schema = idx.schema();
            let fields = fields_from_schema(&schema).ok()?;
            idx.reader().ok().map(|reader| (reader, fields))
        });

        let sqlite = db_path.and_then(|p| Connection::open(p).ok());

        if tantivy.is_none() && sqlite.is_none() {
            return Ok(None);
        }

        Ok(Some(Self {
            reader: tantivy,
            sqlite,
        }))
    }

    pub fn search(
        &self,
        query: &str,
        filters: SearchFilters,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SearchHit>> {
        if let Some((reader, fields)) = &self.reader {
            return self.search_tantivy(reader, fields, query, filters, limit, offset);
        }
        if let Some(conn) = &self.sqlite {
            return self.search_sqlite(conn, query, filters, limit, offset);
        }
        Ok(Vec::new())
    }

    fn search_tantivy(
        &self,
        reader: &IndexReader,
        fields: &crate::search::tantivy::Fields,
        query: &str,
        filters: SearchFilters,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SearchHit>> {
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(searcher.index(), vec![fields.title, fields.content]);

        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        if !query.trim().is_empty() {
            clauses.push((Occur::Must, parser.parse_query(query)?));
        }

        if !filters.agents.is_empty() {
            let terms = filters
                .agents
                .into_iter()
                .map(|agent| {
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(
                            Term::from_field_text(fields.agent, &agent),
                            IndexRecordOption::Basic,
                        )) as Box<dyn Query>,
                    )
                })
                .collect();
            clauses.push((Occur::Must, Box::new(BooleanQuery::new(terms))));
        }

        if !filters.workspaces.is_empty() {
            let terms = filters
                .workspaces
                .into_iter()
                .map(|ws| {
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(
                            Term::from_field_text(fields.workspace, &ws),
                            IndexRecordOption::Basic,
                        )) as Box<dyn Query>,
                    )
                })
                .collect();
            clauses.push((Occur::Must, Box::new(BooleanQuery::new(terms))));
        }

        if filters.created_from.is_some() || filters.created_to.is_some() {
            use std::ops::Bound::{Included, Unbounded};
            let lower = filters
                .created_from
                .map(|v| Included(Term::from_field_i64(fields.created_at, v)))
                .unwrap_or(Unbounded);
            let upper = filters
                .created_to
                .map(|v| Included(Term::from_field_i64(fields.created_at, v)))
                .unwrap_or(Unbounded);
            let range = RangeQuery::new(lower, upper);
            clauses.push((Occur::Must, Box::new(range)));
        }

        let q: Box<dyn Query> = if clauses.is_empty() {
            Box::new(AllQuery)
        } else if clauses.len() == 1 {
            clauses.pop().unwrap().1
        } else {
            Box::new(BooleanQuery::new(clauses))
        };

        let top_docs = searcher.search(&q, &TopDocs::with_limit(limit).and_offset(offset))?;
        let mut hits = Vec::new();
        for (score, addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let title = doc
                .get_first(fields.title)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let content = doc
                .get_first(fields.content)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let agent = doc
                .get_first(fields.agent)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let snippet = content
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or(content.as_str())
                .trim()
                .chars()
                .take(200)
                .collect::<String>();
            let source = doc
                .get_first(fields.source_path)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let workspace = doc
                .get_first(fields.workspace)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            hits.push(SearchHit {
                title,
                snippet,
                score,
                source_path: source,
                agent,
                workspace,
            });
        }
        Ok(hits)
    }

    fn search_sqlite(
        &self,
        conn: &Connection,
        query: &str,
        filters: SearchFilters,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SearchHit>> {
        let mut sql = String::from(
            "SELECT title, content, agent, workspace, source_path, created_at, bm25(fts_messages) AS score
             FROM fts_messages WHERE fts_messages MATCH ?",
        );
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(query.to_string())];

        if !filters.agents.is_empty() {
            let placeholders = (0..filters.agents.len())
                .map(|_| "?".to_string())
                .collect::<Vec<_>>()
                .join(",");
            sql.push_str(&format!(" AND agent IN ({placeholders})"));
            for a in filters.agents {
                params.push(Box::new(a));
            }
        }

        if !filters.workspaces.is_empty() {
            let placeholders = (0..filters.workspaces.len())
                .map(|_| "?".to_string())
                .collect::<Vec<_>>()
                .join(",");
            sql.push_str(&format!(" AND workspace IN ({placeholders})"));
            for w in filters.workspaces {
                params.push(Box::new(w));
            }
        }

        if filters.created_from.is_some() {
            sql.push_str(" AND created_at >= ?");
            params.push(Box::new(filters.created_from.unwrap()));
        }
        if filters.created_to.is_some() {
            sql.push_str(" AND created_at <= ?");
            params.push(Box::new(filters.created_to.unwrap()));
        }

        sql.push_str(" ORDER BY score LIMIT ? OFFSET ?");
        params.push(Box::new(limit as i64));
        params.push(Box::new(offset as i64));

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            rusqlite::params_from_iter(params.iter().map(|b| &**b)),
            |row| {
                let title: String = row.get(0)?;
                let content: String = row.get(1)?;
                let agent: String = row.get(2)?;
                let workspace: String = row.get(3)?;
                let source_path: String = row.get(4)?;
                let score: f32 = row.get::<_, f64>(6)? as f32;
                let snippet = content
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or(content.as_str())
                    .chars()
                    .take(200)
                    .collect();
                Ok(SearchHit {
                    title,
                    snippet,
                    score,
                    source_path,
                    agent,
                    workspace,
                })
            },
        )?;

        let mut hits = Vec::new();
        for row in rows {
            hits.push(row?);
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::{NormalizedConversation, NormalizedMessage, NormalizedSnippet};
    use crate::search::tantivy::TantivyIndex;
    use tempfile::TempDir;

    #[test]
    fn search_returns_results_with_filters_and_pagination() -> Result<()> {
        let dir = TempDir::new()?;
        let mut index = TantivyIndex::open_or_create(dir.path())?;
        let conv = NormalizedConversation {
            agent_slug: "codex".into(),
            external_id: None,
            title: Some("hello world convo".into()),
            workspace: Some(std::path::PathBuf::from("/tmp/workspace")),
            source_path: dir.path().join("rollout-1.jsonl"),
            started_at: Some(1_700_000_000_000),
            ended_at: None,
            metadata: serde_json::json!({}),
            messages: vec![NormalizedMessage {
                idx: 0,
                role: "user".into(),
                author: Some("me".into()),
                created_at: Some(1_700_000_000_000),
                content: "hello rust world".into(),
                extra: serde_json::json!({}),
                snippets: vec![NormalizedSnippet {
                    file_path: None,
                    start_line: None,
                    end_line: None,
                    language: None,
                    snippet_text: None,
                }],
            }],
        };
        index.add_conversation(&conv)?;
        index.commit()?;

        let client = SearchClient::open(dir.path(), None)?.expect("index present");
        let mut filters = SearchFilters::default();
        filters.agents.insert("codex".into());
        filters.workspaces.insert("/tmp/workspace".into());

        let hits = client.search("hello", filters, 10, 0)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].agent, "codex");
        assert!(hits[0].snippet.contains("hello"));
        Ok(())
    }
}

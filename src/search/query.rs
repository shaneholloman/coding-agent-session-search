use std::collections::HashSet;
use std::ops::Bound::{Included, Unbounded};
use std::path::Path;

use anyhow::Result;
use tantivy::collector::TopDocs;
use tantivy::query::{AllQuery, BooleanQuery, Occur, Query, QueryParser, RangeQuery, TermQuery};
use tantivy::schema::{Document, IndexRecordOption, Term};
use tantivy::{Index, IndexReader};

use crate::search::tantivy::fields_from_schema;

#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    pub agents: HashSet<String>,
    pub created_from: Option<i64>,
    pub created_to: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub title: String,
    pub snippet: String,
    pub score: f32,
    pub source_path: String,
}

pub struct SearchClient {
    reader: IndexReader,
    fields: crate::search::tantivy::Fields,
}

impl SearchClient {
    pub fn open(path: &Path) -> Result<Option<Self>> {
        let index = match Index::open_in_dir(path) {
            Ok(idx) => idx,
            Err(_) => return Ok(None),
        };
        let schema = index.schema();
        let fields = fields_from_schema(&schema)?;
        let reader = index.reader()?;
        Ok(Some(Self { reader, fields }))
    }

    pub fn search(
        &self,
        query: &str,
        filters: SearchFilters,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(
            searcher.index(),
            vec![self.fields.title, self.fields.content],
        );

        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        if !query.trim().is_empty() {
            clauses.push((Occur::Must, parser.parse_query(query)?));
        }

        if !filters.agents.is_empty() {
            let mut terms = Vec::new();
            for agent in filters.agents {
                terms.push((
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(self.fields.agent, &agent),
                        IndexRecordOption::Basic,
                    )) as Box<dyn Query>,
                ));
            }
            clauses.push((Occur::Must, Box::new(BooleanQuery::new(terms))));
        }

        if filters.created_from.is_some() || filters.created_to.is_some() {
            let lower = filters
                .created_from
                .map(|v| Included(Term::from_field_i64(self.fields.created_at, v)))
                .unwrap_or(Unbounded);
            let upper = filters
                .created_to
                .map(|v| Included(Term::from_field_i64(self.fields.created_at, v)))
                .unwrap_or(Unbounded);
            let range = RangeQuery::new(lower, upper);
            clauses.push((Occur::Must, Box::new(range)));
        }

        let final_query: Box<dyn Query> = if clauses.is_empty() {
            Box::new(AllQuery)
        } else if clauses.len() == 1 {
            clauses.pop().unwrap().1
        } else {
            Box::new(BooleanQuery::new(clauses))
        };

        let top_docs = searcher.search(&final_query, &TopDocs::with_limit(limit))?;
        let mut hits = Vec::new();
        for (score, addr) in top_docs {
            let doc: Document = searcher.doc(addr)?;
            let title = doc
                .get_first(self.fields.title)
                .and_then(|v| v.as_text())
                .unwrap_or("")
                .to_string();
            let content = doc
                .get_first(self.fields.content)
                .and_then(|v| v.as_text())
                .unwrap_or("")
                .to_string();
            let snippet = content.lines().next().unwrap_or("").to_string();
            let source = doc
                .get_first(self.fields.source_path)
                .and_then(|v| v.as_text())
                .unwrap_or("")
                .to_string();
            hits.push(SearchHit {
                title,
                snippet,
                score,
                source_path: source,
            });
        }
        Ok(hits)
    }
}

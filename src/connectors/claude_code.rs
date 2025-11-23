use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;
use walkdir::WalkDir;

use crate::connectors::{
    Connector, DetectionResult, NormalizedConversation, NormalizedMessage, ScanContext,
};

pub struct ClaudeCodeConnector;
impl Default for ClaudeCodeConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeCodeConnector {
    pub fn new() -> Self {
        Self
    }

    fn projects_root() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_default()
            .join(".claude/projects")
    }
}

impl Connector for ClaudeCodeConnector {
    fn detect(&self) -> DetectionResult {
        let root = Self::projects_root();
        if root.exists() {
            DetectionResult {
                detected: true,
                evidence: vec![format!("found {}", root.display())],
            }
        } else {
            DetectionResult::not_found()
        }
    }

    fn scan(&self, ctx: &ScanContext) -> Result<Vec<NormalizedConversation>> {
        let root = if ctx.data_root.exists() {
            ctx.data_root.clone()
        } else {
            Self::projects_root()
        };
        if !root.exists() {
            return Ok(Vec::new());
        }

        let mut convs = Vec::new();
        for entry in WalkDir::new(&root).into_iter().flatten() {
            if !entry.file_type().is_file() {
                continue;
            }
            let ext = entry.path().extension().and_then(|s| s.to_str());
            if ext != Some("jsonl") && ext != Some("json") && ext != Some("claude") {
                continue;
            }
            let content = fs::read_to_string(entry.path())
                .with_context(|| format!("read {}", entry.path().display()))?;
            let mut messages = Vec::new();
            let mut started_at = None;
            let mut ended_at = None;
            if ext == Some("jsonl") {
                for (idx, line) in content.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let val: Value =
                        serde_json::from_str(line).unwrap_or(Value::String(line.to_string()));
                    let role = val
                        .get("role")
                        .or_else(|| val.get("type"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("agent");
                    let created = val
                        .get("timestamp")
                        .or_else(|| val.get("time"))
                        .and_then(|v| v.as_i64());
                    started_at = started_at.or(created);
                    ended_at = created.or(ended_at);
                    let content_str = val
                        .get("content")
                        .or_else(|| val.get("text"))
                        .and_then(|v| v.as_str())
                        .unwrap_or(line);

                    messages.push(NormalizedMessage {
                        idx: idx as i64,
                        role: role.to_string(),
                        author: None,
                        created_at: created,
                        content: content_str.to_string(),
                        extra: val,
                        snippets: Vec::new(),
                    });
                }
            } else {
                let val: Value = serde_json::from_str(&content).unwrap_or(Value::Null);
                if let Some(arr) = val.get("messages").and_then(|m| m.as_array()) {
                    for (idx, item) in arr.iter().enumerate() {
                        let role = item
                            .get("role")
                            .or_else(|| item.get("type"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("agent");
                        let created = item
                            .get("timestamp")
                            .or_else(|| item.get("time"))
                            .and_then(|v| v.as_i64());
                        started_at = started_at.or(created);
                        ended_at = created.or(ended_at);
                        let content_str = item
                            .get("content")
                            .or_else(|| item.get("text"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        messages.push(NormalizedMessage {
                            idx: idx as i64,
                            role: role.to_string(),
                            author: None,
                            created_at: created,
                            content: content_str.to_string(),
                            extra: item.clone(),
                            snippets: Vec::new(),
                        });
                    }
                }
            }
            if messages.is_empty() {
                continue;
            }
            let title = if ext == Some("jsonl") {
                messages
                    .first()
                    .and_then(|m| m.content.lines().next())
                    .map(|s| s.to_string())
            } else {
                serde_json::from_str::<Value>(&content)
                    .ok()
                    .and_then(|v| {
                        v.get("title")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    })
                    .or_else(|| {
                        messages
                            .first()
                            .and_then(|m| m.content.lines().next())
                            .map(|s| s.to_string())
                    })
            };
            convs.push(NormalizedConversation {
                agent_slug: "claude_code".into(),
                external_id: entry
                    .path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string()),
                title,
                workspace: None,
                source_path: entry.path().to_path_buf(),
                started_at,
                ended_at,
                metadata: serde_json::json!({"source": "claude_code"}),
                messages,
            });
        }

        Ok(convs)
    }
}

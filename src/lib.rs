pub mod config;
pub mod connectors;
pub mod indexer;
pub mod model;
pub mod search;
pub mod storage;
pub mod ui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use indexer::IndexOptions;

/// Command-line interface.
#[derive(Parser, Debug)]
#[command(
    name = "coding-agent-search",
    version,
    about = "Unified TUI search over coding agent histories"
)]
pub struct Cli {
    /// Path to the SQLite database (defaults to platform data dir)
    #[arg(long)]
    pub db: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Launch interactive TUI
    Tui,
    /// Run indexer
    Index {
        /// Perform full rebuild
        #[arg(long)]
        full: bool,

        /// Watch for changes and reindex automatically
        #[arg(long)]
        watch: bool,
    },
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Tui => ui::tui::run_tui(),
        Commands::Index { full, watch } => run_index(cli.db, full, watch),
    }
}

fn run_index(db_override: Option<PathBuf>, full: bool, watch: bool) -> Result<()> {
    let db_path = db_override.unwrap_or_else(default_db_path);
    let data_dir = default_data_dir();
    let opts = IndexOptions {
        full,
        watch,
        db_path,
        data_dir,
    };
    indexer::run_index(opts)
}

fn default_db_path() -> PathBuf {
    default_data_dir().join("agent_search.db")
}

fn default_data_dir() -> PathBuf {
    directories::ProjectDirs::from("com", "coding-agent-search", "coding-agent-search")
        .expect("project dirs available")
        .data_dir()
        .to_path_buf()
}

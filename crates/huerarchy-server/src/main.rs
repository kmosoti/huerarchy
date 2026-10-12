use std::path::PathBuf;

use clap::{Parser, Subcommand};
use huerarchy_core::ObservationFilter;
use huerarchy_server::storage::resolve_database_path;
use huerarchy_server::{HuerarchyServer, Storage};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Parser, Debug)]
#[command(
    name = "huerarchy",
    author = "kmosoti",
    version = env!("CARGO_PKG_VERSION"),
    about = "Huerarchy: Private Rust-native substrate for autonomous AI agents",
    long_about = "Huerarchy provides a shared computational substrate and deterministic interfaces (MCP over stdio) for agent runtimes including Claude Code, Codex, and Antigravity."
)]
struct Cli {
    /// Explicit SQLite database file path.
    #[arg(short = 'd', long, value_name = "PATH", env = "HUERARCHY_DB_PATH")]
    db_path: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run the MCP server over stdio (default behavior)
    Run,

    /// Inspect substrate readiness, schema version, and database status
    Health,

    /// Manage or inspect capabilities
    Capabilities {
        #[command(subcommand)]
        sub: CapabilityCommands,
    },

    /// Inspect durable observations
    Observations {
        #[command(subcommand)]
        sub: ObservationCommands,
    },
}

#[derive(Subcommand, Debug)]
enum CapabilityCommands {
    /// List all registered capability contracts
    List,
}

#[derive(Subcommand, Debug)]
enum ObservationCommands {
    /// Read observations with bounded pagination
    List {
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long)]
        author: Option<String>,
        #[arg(long)]
        subject: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    // MCP protocol isolation: ALL diagnostic logging MUST go to stderr, NEVER stdout!
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    let db_path = resolve_database_path(cli.db_path.as_deref())?;
    let storage = Storage::open(&db_path)?;

    match cli.command.unwrap_or(Commands::Run) {
        Commands::Run => {
            let server = HuerarchyServer::new(storage);
            server.run_stdio().await?;
        }
        Commands::Health => {
            let health = storage.health()?;
            println!("{}", serde_json::to_string_pretty(&health)?);
        }
        Commands::Capabilities { sub } => match sub {
            CapabilityCommands::List => {
                let caps = storage.list_capabilities()?;
                println!("{}", serde_json::to_string_pretty(&caps)?);
            }
        },
        Commands::Observations { sub } => match sub {
            ObservationCommands::List {
                limit,
                offset,
                author,
                subject,
            } => {
                let author_id = match author {
                    Some(a) => Some(huerarchy_core::ParticipantId::new(a)?),
                    None => None,
                };
                let filter = ObservationFilter {
                    author: author_id,
                    subject,
                    since: None,
                    limit,
                    offset,
                };
                let page = storage.read_observations(&filter)?;
                println!("{}", serde_json::to_string_pretty(&page)?);
            }
        },
    }

    Ok(())
}

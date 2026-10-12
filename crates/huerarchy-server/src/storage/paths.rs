use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::storage::error::StorageError;

/// Environment variable for overriding the database path.
pub const HUERARCHY_DB_PATH_ENV: &str = "HUERARCHY_DB_PATH";

/// Resolves the default database path according to environment and system defaults.
pub fn resolve_database_path(explicit_path: Option<&Path>) -> Result<PathBuf, StorageError> {
    if let Some(path) = explicit_path {
        return sanitize_path(path);
    }

    if let Ok(env_path) = env::var(HUERARCHY_DB_PATH_ENV) {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            return sanitize_path(Path::new(trimmed));
        }
    }

    // Default per-user location: ~/.local/share/huerarchy/huerarchy.db
    let base_dir = if let Ok(xdg_data) = env::var("XDG_DATA_HOME") {
        PathBuf::from(xdg_data)
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".local").join("share")
    } else {
        return Err(StorageError::PathError(
            "Neither XDG_DATA_HOME nor HOME environment variable is defined".to_string(),
        ));
    };

    let db_dir = base_dir.join("huerarchy");
    let db_path = db_dir.join("huerarchy.db");
    sanitize_path(&db_path)
}

/// Ensures the directory for the database exists with secure permissions (0700 on Unix).
pub fn ensure_database_dir(path: &Path) -> Result<(), StorageError> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let permissions = fs::Permissions::from_mode(0o700);
                let _ = fs::set_permissions(parent, permissions);
            }
        }
    }
    Ok(())
}

fn sanitize_path(path: &Path) -> Result<PathBuf, StorageError> {
    if path.as_os_str().is_empty() {
        return Err(StorageError::PathError("Path cannot be empty".to_string()));
    }
    Ok(path.to_path_buf())
}

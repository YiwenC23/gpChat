"""
Integration module to automatically update vector database on application startup.
This module provides hooks to integrate vector database updates with your application.
"""

import os
import threading
import logging
from typing import Optional, Callable
from datetime import datetime, timedelta
import time

from vector_db.utils.db_updater import VectorDBUpdater

logger = logging.getLogger('vector_db.app_integration')

# Global state tracking
_updater_thread = None
_last_update_time = None
_update_interval = 3600  # Default: 1 hour in seconds


def initialize_vector_db(
    run_initial_update: bool = True,
    enable_background_updates: bool = True,
    update_interval_seconds: int = 3600,
    batch_size: Optional[int] = None,
    lookback_days: Optional[int] = None,
    on_update_complete: Optional[Callable[[int], None]] = None
):
    """Initialize vector database integration with the application.
    
    This function should be called during application startup to ensure
    the vector database is kept in sync with PostgreSQL data.
    
    Args:
        run_initial_update: Whether to run an update immediately on startup
        enable_background_updates: Whether to enable periodic background updates
        update_interval_seconds: Interval between background updates in seconds
        batch_size: Number of messages to process in each update
        lookback_days: How many days back to look for messages
        on_update_complete: Callback function to execute after each update,
                           receives the number of documents added as argument
    
    Returns:
        True if initialization was successful, False otherwise
    """
    global _update_interval
    
    try:
        # Configure update interval
        _update_interval = update_interval_seconds
        
        # Create updater instance
        updater = VectorDBUpdater(
            batch_size=batch_size,
            lookback_days=lookback_days
        )
        
        # Run initial update if requested
        if run_initial_update:
            logger.info("Running initial vector database update...")
            
            # Run in a separate thread to avoid blocking app startup
            def _run_initial_update():
                try:
                    documents_added = updater.update_vector_db()
                    logger.info(f"Initial update complete. Added {documents_added} documents.")
                    
                    # Update last update timestamp
                    global _last_update_time
                    _last_update_time = datetime.now()
                    
                    # Execute callback if provided
                    if on_update_complete and documents_added is not None:
                        on_update_complete(documents_added)
                        
                except Exception as e:
                    logger.error(f"Error during initial vector database update: {e}")
            
            # Start initial update in background thread
            threading.Thread(
                target=_run_initial_update,
                daemon=True,
                name="vector-db-initial-update"
            ).start()
        
        # Start background updater if enabled
        if enable_background_updates:
            start_background_updater(
                updater=updater,
                update_interval=update_interval_seconds,
                on_update_complete=on_update_complete
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing vector database integration: {e}")
        return False


def start_background_updater(
    updater: Optional[VectorDBUpdater] = None,
    update_interval: Optional[int] = None,
    on_update_complete: Optional[Callable[[int], None]] = None
):
    """Start the background updater thread.
    
    Args:
        updater: VectorDBUpdater instance to use
        update_interval: Interval between updates in seconds
        on_update_complete: Callback function to execute after each update
    """
    global _updater_thread, _update_interval
    
    # Don't start if already running
    if _updater_thread and _updater_thread.is_alive():
        logger.info("Background updater already running")
        return
    
    # Use provided interval or default
    interval = update_interval or _update_interval
    
    # Create updater if not provided
    if updater is None:
        updater = VectorDBUpdater()
    
    # Background update function
    def _background_updater():
        logger.info(f"Starting background vector database updater (interval: {interval}s)")
        
        while True:
            try:
                # Sleep first to avoid running update immediately after initial update
                time.sleep(interval)
                
                # Run the update
                logger.info("Running scheduled vector database update...")
                documents_added = updater.update_vector_db()
                
                # Update timestamp
                global _last_update_time
                _last_update_time = datetime.now()
                
                # Log result
                if documents_added is not None:
                    logger.info(f"Scheduled update complete. Added {documents_added} documents.")
                    
                    # Execute callback if provided
                    if on_update_complete:
                        on_update_complete(documents_added)
                else:
                    logger.warning("Scheduled update completed but may have encountered errors.")
                    
            except Exception as e:
                logger.error(f"Error in background updater: {e}")
    
    # Start background thread
    _updater_thread = threading.Thread(
        target=_background_updater,
        daemon=True,
        name="vector-db-background-updater"
    )
    _updater_thread.start()


def get_last_update_time() -> Optional[datetime]:
    """Get the timestamp of the last vector database update."""
    return _last_update_time


def is_updater_running() -> bool:
    """Check if the background updater is currently running."""
    return _updater_thread is not None and _updater_thread.is_alive()


# Example usage in your application's main module:
"""
from vector_db.app_integration import initialize_vector_db

def on_vector_db_update(docs_added):
    print(f"Vector database updated with {docs_added} new documents")

def main():
    # Application initialization...
    
    # Initialize vector database integration
    initialize_vector_db(
        run_initial_update=True,
        enable_background_updates=True,
        update_interval_seconds=3600,
        lookback_days=1,
        on_update_complete=on_vector_db_update
    )
    
    # Rest of application startup...
""" 
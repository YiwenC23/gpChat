#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv
from prettytable import PrettyTable

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create and return a database connection."""
    return psycopg2.connect(
        dbname=os.getenv('POSTGRES_DB', 'zulip'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432')
    )

def list_tables(conn):
    """List all tables in the database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        return [row[0] for row in cur.fetchall()]

def inspect_table(conn, table_name):
    """Inspect and display data from a specific table."""
    with conn.cursor() as cur:
        # Get column names
        cur.execute(f"SELECT * FROM {table_name} LIMIT 0")
        columns = [desc[0] for desc in cur.description]
        
        # Get data
        cur.execute(f"SELECT * FROM {table_name} LIMIT 100")  # Limit to 100 rows
        rows = cur.fetchall()
        
        # Create and print table
        table = PrettyTable(columns)
        for row in rows:
            # Truncate long text fields for better display
            table.add_row([str(cell)[:50] + '...' if cell and len(str(cell)) > 50 else cell for cell in row])
        
        print(f"\nTable: {table_name}")
        print(f"Showing {len(rows)} rows\n{table}\n")

def main():
    try:
        conn = get_db_connection()
        print("Successfully connected to PostgreSQL database!")
        
        tables = list_tables(conn)
        print("\nAvailable tables:")
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")
        
        while True:
            try:
                choice = input("\nEnter table number to inspect (q to quit): ").strip().lower()
                if choice == 'q':
                    break
                
                table_num = int(choice) - 1
                if 0 <= table_num < len(tables):
                    inspect_table(conn, tables[table_num])
                else:
                    print("Invalid table number. Please try again.")
            
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                break
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
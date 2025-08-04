#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv
from prettytable import PrettyTable
import numpy as np

# Load environment variables
load_dotenv()

class VectorDBInspector:
    def __init__(self):
        self.conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        # No register_vector needed
    
    def list_vector_tables(self):
        """List all tables that have vector columns."""
        with self.conn.cursor() as cur:
            # Look for tables with vector columns
            cur.execute("""
                SELECT table_name, column_name 
                FROM information_schema.columns 
                WHERE udt_name = 'vector'
                ORDER BY table_name;
            """)
            return cur.fetchall()
    
    def inspect_vector_table(self, table_name, vector_column='embedding'):
        """Inspect a table with vector data."""
        with self.conn.cursor() as cur:
            # Get column info
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            columns = [col[0] for col in cur.fetchall()]
            
            # Get sample data (limit to 10 rows for display)
            cur.execute(f"SELECT * FROM {table_name} LIMIT 10")
            rows = cur.fetchall()
            
            # Create and print table
            table = PrettyTable(columns)
            for row in rows:
                # Format the row data
                formatted_row = []
                for i, cell in enumerate(row):
                    if columns[i] == vector_column and cell is not None:
                        # For vector columns, just show the type and length
                        formatted_row.append(f"Vector[{len(cell) if hasattr(cell, '__len__') else '?'}]")
                    else:
                        formatted_row.append(str(cell)[:50] + '...' if cell and len(str(cell)) > 50 else cell)
                table.add_row(formatted_row)
            
            print(f"\nTable: {table_name}")
            print(f"Showing {len(rows)} rows\n{table}")
            
            # Show vector stats if available
            if vector_column in columns:
                cur.execute(f"""
                    SELECT 
                        COUNT(*) as total_rows
                    FROM {table_name};
                """)
                stats = cur.fetchone()
                print("\nVector Statistics:")
                print(f"Total vectors: {stats[0]}")
    
    def close(self):
        self.conn.close()

def main():
    try:
        inspector = VectorDBInspector()
        print("Connected to Vector Database!")
        
        vector_tables = inspector.list_vector_tables()
        if not vector_tables:
            print("No vector tables found in the database.")
            return
        
        print("\nTables with vector columns:")
        for i, (table, column) in enumerate(vector_tables, 1):
            print(f"{i}. {table}.{column}")
        
        while True:
            try:
                choice = input("\nEnter table number to inspect (q to quit): ").strip().lower()
                if choice == 'q':
                    break
                
                table_num = int(choice) - 1
                if 0 <= table_num < len(vector_tables):
                    table_name, vector_column = vector_tables[table_num]
                    inspector.inspect_vector_table(table_name, vector_column)
                else:
                    print("Invalid table number. Please try again.")
            
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                break
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'inspector' in locals():
            inspector.close()

if __name__ == "__main__":
    main()
import csv
from typing import List

from ..models import Table


class CSVExporter:
    """Export tables to CSV format."""

    def __init__(self, delimiter: str = ";", encoding: str = "utf-8-sig"):
        self.delimiter = delimiter
        self.encoding = encoding

    def export(
        self,
        tables: List[Table],
        output_path: str,
        include_headers: bool = True,
        skip_empty: bool = False,
    ):
        """Export tables to CSV file(s).

        If multiple tables, only the first one is exported to the given path.
        Use export_all for multiple files.
        """
        if not tables:
            return

        self._export_table(
            tables[0], output_path,
            include_headers=include_headers,
            skip_empty=skip_empty
        )

    def export_all(
        self,
        tables: List[Table],
        output_dir: str,
        base_name: str = "table",
        include_headers: bool = True,
        skip_empty: bool = False,
    ) -> List[str]:
        """Export all tables to separate CSV files.

        Returns list of created file paths.
        """
        import os

        created_files = []

        for i, table in enumerate(tables):
            file_name = f"{base_name}_{table.name}.csv"
            file_path = os.path.join(output_dir, file_name)

            self._export_table(
                table, file_path,
                include_headers=include_headers,
                skip_empty=skip_empty
            )
            created_files.append(file_path)

        return created_files

    def _export_table(
        self,
        table: Table,
        output_path: str,
        include_headers: bool,
        skip_empty: bool,
    ):
        """Export a single table to CSV."""
        with open(output_path, "w", newline="", encoding=self.encoding) as f:
            writer = csv.writer(f, delimiter=self.delimiter, quoting=csv.QUOTE_MINIMAL)

            # Get enabled columns
            enabled_cols = [col for col in table.columns if col.enabled]
            if not enabled_cols:
                return

            # Write headers
            if include_headers:
                headers = [col.name for col in enabled_cols]
                writer.writerow(headers)

            # Write data
            data_matrix = table.get_data_matrix(enabled_only=True)

            for row_data in data_matrix:
                # Skip empty rows if requested
                if skip_empty and all(not cell.strip() for cell in row_data):
                    continue

                writer.writerow(row_data)

    def to_string(
        self,
        table: Table,
        include_headers: bool = True,
        skip_empty: bool = False,
    ) -> str:
        """Export table to CSV string."""
        import io

        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.delimiter, quoting=csv.QUOTE_MINIMAL)

        enabled_cols = [col for col in table.columns if col.enabled]
        if not enabled_cols:
            return ""

        if include_headers:
            headers = [col.name for col in enabled_cols]
            writer.writerow(headers)

        data_matrix = table.get_data_matrix(enabled_only=True)

        for row_data in data_matrix:
            if skip_empty and all(not cell.strip() for cell in row_data):
                continue
            writer.writerow(row_data)

        return output.getvalue()

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pymysql

from app.config import Settings


@dataclass(slots=True)
class ParseRunRecord:
    source_file: str
    dataset: str
    status: str
    warning_count: int
    extracted_rows: int
    parser_method: str
    note: str = ""


class AdmissionsRepository:
    def __init__(self, settings: Settings):
        self.settings = settings

    @contextmanager
    def connect(self):
        connection = pymysql.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            user=self.settings.mysql_user,
            password=self.settings.mysql_password,
            database=self.settings.mysql_database,
            charset="utf8mb4",
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def ensure_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                dataset VARCHAR(64) NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                source_doc VARCHAR(255) NOT NULL,
                title VARCHAR(255) NOT NULL,
                source_type VARCHAR(32) NOT NULL,
                source_path TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                UNIQUE KEY uniq_documents_dataset_file (dataset, source_file)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS parse_runs (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                source_file VARCHAR(255) NOT NULL,
                dataset VARCHAR(64) NOT NULL,
                status VARCHAR(32) NOT NULL,
                warning_count INT NOT NULL DEFAULT 0,
                extracted_rows INT NOT NULL DEFAULT 0,
                parser_method VARCHAR(64) NOT NULL,
                note TEXT NOT NULL,
                created_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS major_catalog (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                academic_year VARCHAR(16) NOT NULL,
                source_dataset VARCHAR(64) NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                source_doc VARCHAR(255) NOT NULL,
                source_table_title VARCHAR(255) NOT NULL,
                source_row_no VARCHAR(32) NOT NULL,
                major_code VARCHAR(64) NOT NULL,
                major_name VARCHAR(255) NOT NULL,
                duration VARCHAR(64) NOT NULL,
                tuition VARCHAR(64) NOT NULL,
                exam_subjects VARCHAR(255) NOT NULL,
                degree_type VARCHAR(128) NOT NULL,
                college_name VARCHAR(255) NOT NULL,
                evidence_text TEXT NOT NULL,
                extract_time VARCHAR(32) NOT NULL,
                created_at DATETIME NOT NULL,
                KEY idx_major_catalog_name (major_name),
                KEY idx_major_catalog_code (major_code),
                KEY idx_major_catalog_college (college_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS score_lines (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                source_dataset VARCHAR(64) NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                source_sheet VARCHAR(255) NOT NULL,
                source_row_no VARCHAR(32) NOT NULL,
                year VARCHAR(16) NOT NULL,
                province VARCHAR(64) NOT NULL,
                batch VARCHAR(128) NOT NULL,
                category VARCHAR(128) NOT NULL,
                major_name VARCHAR(255) NOT NULL,
                min_score VARCHAR(64) NOT NULL,
                min_rank VARCHAR(64) NOT NULL,
                evidence_text TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                KEY idx_score_lines_year (year),
                KEY idx_score_lines_province (province),
                KEY idx_score_lines_batch (batch),
                KEY idx_score_lines_major (major_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS policy_tables (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                source_dataset VARCHAR(64) NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                source_doc VARCHAR(255) NOT NULL,
                table_topic VARCHAR(255) NOT NULL,
                source_row_no VARCHAR(32) NOT NULL,
                field_name VARCHAR(128) NOT NULL,
                field_value TEXT NOT NULL,
                evidence_text TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                KEY idx_policy_tables_topic (table_topic),
                KEY idx_policy_tables_doc (source_doc)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS faq_seed (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                source_file VARCHAR(255) NOT NULL,
                tag_name VARCHAR(255) NOT NULL,
                question_no VARCHAR(32) NOT NULL,
                question TEXT NOT NULL,
                answer LONGTEXT NOT NULL,
                retrieval_priority VARCHAR(32) NOT NULL,
                created_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        ]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)

    def replace_major_catalog(self, rows: list[dict[str, str]]) -> int:
        now = datetime.now()
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM major_catalog")
                if not rows:
                    return 0
                cursor.executemany(
                    """
                    INSERT INTO major_catalog (
                        academic_year, source_dataset, source_file, source_doc, source_table_title, source_row_no,
                        major_code, major_name, duration, tuition, exam_subjects, degree_type, college_name,
                        evidence_text, extract_time, created_at
                    ) VALUES (
                        %(academic_year)s, %(source_dataset)s, %(source_file)s, %(source_doc)s, %(source_table_title)s, %(source_row_no)s,
                        %(major_code)s, %(major_name)s, %(duration)s, %(tuition)s, %(exam_subjects)s, %(degree_type)s, %(college_name)s,
                        %(evidence_text)s, %(extract_time)s, %(created_at)s
                    )
                    """,
                    [{**row, "created_at": now} for row in rows],
                )
        return len(rows)

    def replace_score_lines(self, rows: list[dict[str, str]]) -> int:
        now = datetime.now()
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM score_lines")
                if not rows:
                    return 0
                cursor.executemany(
                    """
                    INSERT INTO score_lines (
                        source_dataset, source_file, source_sheet, source_row_no, year, province, batch,
                        category, major_name, min_score, min_rank, evidence_text, created_at
                    ) VALUES (
                        %(source_dataset)s, %(source_file)s, %(source_sheet)s, %(source_row_no)s, %(year)s, %(province)s, %(batch)s,
                        %(category)s, %(major_name)s, %(min_score)s, %(min_rank)s, %(evidence_text)s, %(created_at)s
                    )
                    """,
                    [{**row, "created_at": now} for row in rows],
                )
        return len(rows)

    def replace_policy_tables(self, rows: list[dict[str, str]]) -> int:
        now = datetime.now()
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM policy_tables")
                if not rows:
                    return 0
                cursor.executemany(
                    """
                    INSERT INTO policy_tables (
                        source_dataset, source_file, source_doc, table_topic, source_row_no, field_name, field_value, evidence_text, created_at
                    ) VALUES (
                        %(source_dataset)s, %(source_file)s, %(source_doc)s, %(table_topic)s, %(source_row_no)s, %(field_name)s, %(field_value)s, %(evidence_text)s, %(created_at)s
                    )
                    """,
                    [{**row, "created_at": now} for row in rows],
                )
        return len(rows)

    def replace_faq_seed(self, rows: list[dict[str, str]]) -> int:
        now = datetime.now()
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM faq_seed")
                if not rows:
                    return 0
                cursor.executemany(
                    """
                    INSERT INTO faq_seed (
                        source_file, tag_name, question_no, question, answer, retrieval_priority, created_at
                    ) VALUES (
                        %(source_file)s, %(tag_name)s, %(question_no)s, %(question)s, %(answer)s, %(retrieval_priority)s, %(created_at)s
                    )
                    """,
                    [{**row, "created_at": now} for row in rows],
                )
        return len(rows)

    def replace_documents(self, rows: list[dict[str, str]]) -> int:
        now = datetime.now()
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM documents")
                if not rows:
                    return 0
                cursor.executemany(
                    """
                    INSERT INTO documents (
                        dataset, source_file, source_doc, title, source_type, source_path, created_at
                    ) VALUES (
                        %(dataset)s, %(source_file)s, %(source_doc)s, %(title)s, %(source_type)s, %(source_path)s, %(created_at)s
                    )
                    """,
                    [{**row, "created_at": now} for row in rows],
                )
        return len(rows)

    def append_parse_runs(self, rows: list[ParseRunRecord]) -> int:
        now = datetime.now()
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO parse_runs (
                        source_file, dataset, status, warning_count, extracted_rows, parser_method, note, created_at
                    ) VALUES (
                        %(source_file)s, %(dataset)s, %(status)s, %(warning_count)s, %(extracted_rows)s, %(parser_method)s, %(note)s, %(created_at)s
                    )
                    """,
                    [
                        {
                            "source_file": row.source_file,
                            "dataset": row.dataset,
                            "status": row.status,
                            "warning_count": row.warning_count,
                            "extracted_rows": row.extracted_rows,
                            "parser_method": row.parser_method,
                            "note": row.note,
                            "created_at": now,
                        }
                        for row in rows
                    ],
                )
        return len(rows)


def flatten_policy_table_rows(table_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for row in table_rows:
        evidence_text = "；".join(
            f"{label}：{row.get(label, '')}"
            for label in ("专业代码", "专业名称", "学制", "学费（元）", "选考科目", "学位授予门类", "所在院系")
            if row.get(label)
        )
        for field_name, field_value in row.items():
            if field_name.startswith("source_") or field_name == "extract_time" or field_name == "序号":
                continue
            flattened.append(
                {
                    "source_dataset": "policy_tables",
                    "source_file": row.get("source_file", ""),
                    "source_doc": row.get("source_doc", ""),
                    "table_topic": row.get("source_table_title", ""),
                    "source_row_no": row.get("source_row_no", ""),
                    "field_name": field_name,
                    "field_value": field_value,
                    "evidence_text": evidence_text,
                }
            )
    return flattened

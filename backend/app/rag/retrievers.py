from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
import re
from threading import Lock

from langchain_core.documents import Document

from app.rag.index import RagIndexManager


@dataclass(slots=True)
class RetrievedItem:
    """统一召回结果结构，保留文档对象与融合分数。"""

    document: Document
    score: float


class HybridRetriever:
    """基于 BM25 + Dense + Exact Match + RRF 的混合召回器。"""

    _SEMANTIC_QUERY_MARKERS = (
        "主要介绍了什么",
        "怎么说明",
        "哪些专业相关内容",
        "主要怎么规定",
    )
    _CONSTRAINT_QUERY_MARKERS = (
        "多少",
        "是什么",
        "在哪里",
        "咨询电话",
        "联系方式",
        "官网",
        "官方网站",
        "地址",
        "学费",
        "住宿费",
        "选考科目",
        "录取要求",
    )
    _DOMAIN_KEYWORDS = (
        "学校概况",
        "学校简介",
        "招生政策要点",
        "普通本科招生章程",
        "报考指南",
        "校区与联系方式",
        "历史沿革",
        "学院与专业概览",
        "学生生活与服务",
        "联系方式",
        "咨询电话",
        "地址",
        "学费",
        "住宿费",
        "录取规则",
        "录取要求",
        "奖学金",
        "资助",
        "助学贷款",
        "专业",
        "专业要求",
        "选考科目",
        "学院",
        "网址",
        "国标代码",
        "河南招生代码",
    )

    def __init__(
        self,
        index: RagIndexManager,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        exact_weight: float = 1.2,
        rrf_k: int = 60,
    ):
        self.index = index
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.exact_weight = exact_weight
        self.rrf_k = rrf_k
        self._sparse_lock = Lock()

    def retrieve(self, queries: list[str], top_n: int, focus_parent_ids: list[str] | None = None) -> list[RetrievedItem]:
        """执行多查询召回并融合分数后返回 top_n。"""
        merged_scores: dict[str, float] = {}
        merged_docs: dict[str, Document] = {}
        focus_parent_ids = focus_parent_ids or []
        normalized_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))

        if len(normalized_queries) <= 1:
            query_batches = [
                self._retrieve_single_query(
                    query=normalized_queries[0],
                    top_n=top_n,
                    focus_parent_ids=focus_parent_ids,
                )
            ] if normalized_queries else []
        else:
            max_workers = min(len(normalized_queries), 4)
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-query") as executor:
                futures = [
                    executor.submit(
                        self._retrieve_single_query,
                        query=query,
                        top_n=top_n,
                        focus_parent_ids=focus_parent_ids,
                    )
                    for query in normalized_queries
                ]
                query_batches = [future.result() for future in futures]

        for rows in query_batches:
            for item in rows:
                chunk_id = str(item.document.metadata.get("chunk_id", ""))
                if not chunk_id:
                    continue
                merged_docs[chunk_id] = item.document
                merged_scores[chunk_id] = max(item.score, merged_scores.get(chunk_id, 0.0))

        ranked = [
            RetrievedItem(document=merged_docs[chunk_id], score=score)
            for chunk_id, score in merged_scores.items()
        ]
        ranked.sort(key=lambda row: row.score, reverse=True)
        return ranked[:top_n]

    def _retrieve_single_query(self, query: str, top_n: int, focus_parent_ids: list[str]) -> list[RetrievedItem]:
        """执行单查询召回，并使用 RRF 融合三路排序。"""
        shape = self._classify_query_shape(query)
        local_focus_parent_ids = list(dict.fromkeys([*focus_parent_ids, *self.locate_summary_parents(query, top_n=3)]))
        sparse_weight, dense_weight, exact_weight, rrf_k = self._resolve_query_profile(shape)
        dense_rows = self.index.dense_similarity_scores(query=query, top_k=top_n)
        dense_docs = [item[0] for item in dense_rows]
        exact_docs = self._exact_match_search(query=query, top_n=top_n)
        sparse_docs = self._retrieve_sparse_docs(query=query, top_n=top_n)
        parent_seed_docs = self._collect_parent_seed_docs(
            query=query,
            parent_ids=local_focus_parent_ids,
            top_n=max(top_n, 6),
        )

        scores = self._build_rank_score_map(
            sparse_docs=sparse_docs,
            dense_docs=dense_docs,
            exact_docs=exact_docs,
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
            exact_weight=exact_weight,
            rrf_k=rrf_k,
        )
        merged = self._merge_ranked_docs(
            candidates=[parent_seed_docs, dense_docs, exact_docs, sparse_docs],
            scores=scores,
            query=query,
            top_n=top_n,
        )
        ranked_items: list[RetrievedItem] = []
        for doc in merged:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            score = (
                scores.get(chunk_id, 0.0)
                + self._temporal_bonus(query=query, doc=doc)
                + self._parent_focus_bonus(doc, local_focus_parent_ids)
                + self._query_intent_bonus(query=query, doc=doc)
            )
            ranked_items.append(RetrievedItem(document=doc, score=score))
        return ranked_items

    def _retrieve_sparse_docs(self, query: str, top_n: int) -> list[Document]:
        """BM25 共享实例带状态，查询并发时用锁保护。"""
        with self._sparse_lock:
            bm25 = self.index.get_bm25_retriever(top_k=top_n)
            return bm25.invoke(query)

    def locate_summary_parents(self, query: str, top_n: int = 3) -> list[str]:
        """先检索摘要层，返回应重点关注的 parent_id。"""
        summary_docs = [doc for doc in self.index.all_documents() if str(doc.metadata.get("chunk_level", "")) == "summary"]
        if not summary_docs:
            return []
        tokens = self._tokenize(query)
        scored: list[tuple[float, str]] = []
        for doc in summary_docs:
            text = " ".join(
                [
                    str(doc.metadata.get("section_hint", "")),
                    str(doc.metadata.get("topic", "")),
                    str(doc.metadata.get("chunk_text", "")),
                    doc.page_content,
                ]
            ).lower()
            token_hits = sum(1 for token in tokens if token in text)
            if token_hits <= 0:
                continue
            parent_id = str(doc.metadata.get("parent_id", ""))
            if not parent_id:
                continue
            score = token_hits / max(len(tokens), 1)
            if any(token.isdigit() and token in text for token in tokens):
                score += 0.2
            scored.append((score, parent_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        seen: set[str] = set()
        ordered: list[str] = []
        for _, parent_id in scored:
            if parent_id in seen:
                continue
            seen.add(parent_id)
            ordered.append(parent_id)
            if len(ordered) >= top_n:
                break
        return ordered

    def _build_rank_score_map(
        self,
        sparse_docs: list[Document],
        dense_docs: list[Document],
        exact_docs: list[Document],
        *,
        sparse_weight: float,
        dense_weight: float,
        exact_weight: float,
        rrf_k: int,
    ) -> dict[str, float]:
        """使用 RRF 把多路排序位置映射为可比较分值。"""
        sparse_rank = self._rank_map(sparse_docs)
        dense_rank = self._rank_map(dense_docs)
        exact_rank = self._rank_map(exact_docs)
        scores: dict[str, float] = {}
        for chunk_id in set(sparse_rank) | set(dense_rank) | set(exact_rank):
            score = 0.0
            if chunk_id in sparse_rank:
                score += sparse_weight / (rrf_k + sparse_rank[chunk_id] + 1)
            if chunk_id in dense_rank:
                score += dense_weight / (rrf_k + dense_rank[chunk_id] + 1)
            if chunk_id in exact_rank:
                score += exact_weight / (rrf_k + exact_rank[chunk_id] + 1)
            scores[chunk_id] = score
        return scores

    def _rank_map(self, docs: list[Document]) -> dict[str, int]:
        """生成 chunk_id 到排名位置的映射。"""
        mapping: dict[str, int] = {}
        for idx, doc in enumerate(docs):
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if chunk_id and chunk_id not in mapping:
                mapping[chunk_id] = idx
        return mapping

    def _merge_ranked_docs(self, candidates: list[list[Document]], scores: dict[str, float], query: str, top_n: int) -> list[Document]:
        """按融合分数合并候选文档，并做基础时效过滤与排序。"""
        seen: set[str] = set()
        ordered_candidates: list[Document] = []
        for rows in candidates:
            for row in rows:
                chunk_id = str(row.metadata.get("chunk_id", ""))
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                ordered_candidates.append(row)
        ordered_candidates = self._apply_temporal_policy(query=query, docs=ordered_candidates)
        ordered_candidates.sort(
            key=lambda doc: (
                scores.get(str(doc.metadata.get("chunk_id", "")), 0.0)
                + self._temporal_bonus(query=query, doc=doc)
                + self._query_intent_bonus(query=query, doc=doc),
                str(doc.metadata.get("publish_date", "")),
            ),
            reverse=True,
        )
        return ordered_candidates[:top_n]

    def _exact_match_search(self, query: str, top_n: int) -> list[Document]:
        """关键词精确匹配通道，补足对金额、年份、条款号等硬约束的召回。"""
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scored: list[tuple[float, Document]] = []
        normalized_query = query.strip().lower()
        for doc in self.index.all_documents():
            query_expansions = doc.metadata.get("query_expansions", [])
            expansion_text = " ".join(str(item) for item in query_expansions) if isinstance(query_expansions, list) else ""
            raw_text = " ".join(
                [
                    str(doc.metadata.get("chunk_text") or ""),
                    str(doc.metadata.get("source_title") or ""),
                    str(doc.metadata.get("topic") or ""),
                    str(doc.metadata.get("section_hint") or ""),
                    expansion_text,
                    str(doc.page_content),
                ]
            )
            text = raw_text.lower()
            phrase_hit = 1.5 if normalized_query and normalized_query in text else 0.0
            token_hits = sum(1 for token in tokens if token in text)
            if token_hits <= 0 and phrase_hit <= 0:
                continue
            coverage = token_hits / max(len(tokens), 1)
            numeric_hits = sum(1 for token in tokens if token.isdigit() and token in text)
            field_hit_bonus = self._field_alignment_bonus(query=query, doc=doc)
            score = phrase_hit + coverage + numeric_hits * 0.25 + field_hit_bonus + self._query_intent_bonus(query=query, doc=doc)
            cloned = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            cloned.metadata["score"] = score
            scored.append((score, cloned))
        scored.sort(key=lambda item: (item[0], str(item[1].metadata.get("publish_date", ""))), reverse=True)
        return [item[1] for item in scored[:top_n]]

    def _tokenize(self, text: str) -> list[str]:
        """中英混合分词，供精确匹配通道复用。"""
        lowered = text.lower()
        tokens: list[str] = []
        for raw in re.findall(r"[a-zA-Z]+|\d+|[\u4e00-\u9fff]{2,}", lowered):
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", raw):
                tokens.extend(self._expand_chinese_token(raw))
            else:
                tokens.append(raw)
        return list(dict.fromkeys(tok for tok in tokens if tok))

    def _expand_chinese_token(self, raw: str) -> list[str]:
        """展开中文短语，兼顾原词、切词结果与短 n-gram。"""
        cleaned = raw.strip().lower()
        if len(cleaned) < 2:
            return [cleaned] if cleaned else []

        expanded: list[str] = [cleaned]
        fragments = [part for part in re.split(r"[的了和与及在中对把将就按关于主要什么多少哪里哪些怎么是否请问]", cleaned) if len(part) >= 2]
        expanded.extend(fragments)
        try:
            import jieba

            expanded.extend(token.strip().lower() for token in jieba.lcut(cleaned) if len(token.strip()) >= 2)
        except Exception:
            pass

        generated = 0
        max_window = min(4, len(cleaned))
        for window in range(max_window, 1, -1):
            for start in range(0, len(cleaned) - window + 1):
                expanded.append(cleaned[start : start + window])
                generated += 1
                if generated >= 48:
                    break
            if generated >= 48:
                break
        return list(dict.fromkeys(token for token in expanded if token))

    def _classify_query_shape(self, query: str) -> str:
        """根据问句形态区分语义型、约束型和词面型问题。"""
        normalized = query.strip()
        if not normalized:
            return "lexical"
        if any(marker in normalized for marker in self._SEMANTIC_QUERY_MARKERS):
            return "semantic"
        if any(marker in normalized for marker in self._CONSTRAINT_QUERY_MARKERS):
            return "constraint"
        if any(char.isdigit() for char in normalized):
            return "constraint"
        if len(normalized) <= 14 and not re.search(r"[？?。；!！]", normalized):
            return "lexical"
        if sum(keyword in normalized for keyword in self._DOMAIN_KEYWORDS) >= 2:
            return "lexical"
        return "semantic"

    def _resolve_query_profile(self, shape: str) -> tuple[float, float, float, int]:
        """按问题类型动态调整三路权重和 RRF 平滑系数。"""
        if shape == "constraint":
            return (
                round(self.sparse_weight * 1.0, 4),
                round(self.dense_weight * 0.8, 4),
                round(self.exact_weight * 1.65, 4),
                max(18, self.rrf_k // 2),
            )
        if shape == "lexical":
            return (
                round(self.sparse_weight * 1.15, 4),
                round(self.dense_weight * 0.55, 4),
                round(self.exact_weight * 2.0, 4),
                max(12, self.rrf_k // 3),
            )
        return (
            round(self.sparse_weight * 0.75, 4),
            round(self.dense_weight * 1.45, 4),
            round(self.exact_weight * 0.7, 4),
            max(72, self.rrf_k),
        )

    def _collect_parent_seed_docs(self, query: str, parent_ids: list[str], top_n: int) -> list[Document]:
        """从摘要定位到的 parent 中优先补入候选，减少混合召回漏召。"""
        normalized_parent_ids = list(dict.fromkeys(parent_id for parent_id in parent_ids if parent_id))
        if not normalized_parent_ids or top_n <= 0:
            return []

        parent_id_set = set(normalized_parent_ids)
        tokens = self._tokenize(query)
        scored: list[tuple[float, Document]] = []
        for doc in self.index.all_documents():
            parent_id = str(doc.metadata.get("parent_id", ""))
            if parent_id not in parent_id_set:
                continue
            text = " ".join(
                [
                    str(doc.metadata.get("section_hint", "")),
                    str(doc.metadata.get("topic", "")),
                    str(doc.metadata.get("source_title", "")),
                    str(doc.metadata.get("chunk_text", "")),
                    doc.page_content,
                ]
            ).lower()
            token_hits = sum(1 for token in tokens if token in text)
            coverage = token_hits / max(len(tokens), 1)
            score = coverage + self._field_alignment_bonus(query=query, doc=doc) + self._query_intent_bonus(query=query, doc=doc)
            if str(doc.metadata.get("chunk_level", "")) == "small":
                score += 0.18
            elif str(doc.metadata.get("chunk_level", "")) == "summary":
                score += 0.06
            if token_hits <= 0 and score < 0.2:
                continue
            cloned = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            cloned.metadata["score"] = score
            scored.append((score, cloned))
        scored.sort(key=lambda item: (item[0], str(item[1].metadata.get("publish_date", ""))), reverse=True)
        return [item[1] for item in scored[:top_n]]

    def _field_alignment_bonus(self, query: str, doc: Document) -> float:
        """对字段型问题做弱监督对齐，降低答非所问的候选排序。"""
        intent_aliases = {
            "学费": ("学费", "收费", "培养费"),
            "住宿费": ("住宿费", "住宿标准", "住宿安排"),
            "电话": ("电话", "联系方式", "咨询热线", "咨询电话"),
            "网址": ("网址", "官网", "官方网站", "网站", "http://", "https://"),
            "地址": ("地址", "校区", "通讯地址", "联系地址"),
            "录取": ("录取", "调档", "投档"),
            "选考": ("选考", "选科", "科目要求"),
            "代码": ("国标代码", "招生代码", "院校代码"),
            "资助": ("资助", "奖学金", "助学贷款", "贷款"),
        }
        metadata_text = " ".join(
            [
                str(doc.metadata.get("section_hint", "")),
                str(doc.metadata.get("topic", "")),
                str(doc.metadata.get("source_title", "")),
                str(doc.metadata.get("chunk_text", "")),
            ]
        )
        score = 0.0
        for intent, aliases in intent_aliases.items():
            if intent not in query and not any(alias in query for alias in aliases):
                continue
            hits = sum(1 for alias in aliases if alias in metadata_text)
            if hits <= 0:
                continue
            score += min(0.9, 0.28 + 0.12 * hits)
        return round(score, 4)

    def _apply_temporal_policy(self, query: str, docs: list[Document]) -> list[Document]:
        """对时效问题优先保留年份匹配或当前有效的证据。"""
        if not docs:
            return []
        query_years = set(re.findall(r"\b(20\d{2})\b", query))
        time_sensitive = self._is_time_sensitive(query)
        if not time_sensitive and not query_years:
            return docs

        if query_years:
            matched = [doc for doc in docs if self._doc_matches_year(doc, query_years)]
            if matched:
                return matched + [doc for doc in docs if doc not in matched]

        active_docs = [doc for doc in docs if self._is_active(doc)]
        if active_docs:
            return active_docs + [doc for doc in docs if doc not in active_docs]
        return docs

    def _temporal_bonus(self, query: str, doc: Document) -> float:
        """为时效匹配和有效期命中的文档增加排序偏置。"""
        query_years = set(re.findall(r"\b(20\d{2})\b", query))
        bonus = 0.0
        if query_years and self._doc_matches_year(doc, query_years):
            bonus += 1.2
        if self._is_time_sensitive(query) and self._is_active(doc):
            bonus += 0.6
        if self._is_expired(doc):
            bonus -= 0.8
        return bonus

    def _doc_matches_year(self, doc: Document, years: set[str]) -> bool:
        haystack = " ".join(
            [
                str(doc.metadata.get("publish_date", "")),
                str(doc.metadata.get("effective_date", "")),
                str(doc.metadata.get("expire_date", "")),
                str(doc.metadata.get("source_title", "")),
                str(doc.metadata.get("topic", "")),
                str(doc.metadata.get("chunk_text", "")),
            ]
        )
        return any(year in haystack for year in years)

    def _parent_focus_bonus(self, doc: Document, focus_parent_ids: list[str]) -> float:
        if not focus_parent_ids:
            return 0.0
        parent_id = str(doc.metadata.get("parent_id", ""))
        if parent_id and parent_id in set(focus_parent_ids):
            return 0.9
        return -0.05 if str(doc.metadata.get("chunk_level", "")) == "summary" else 0.0

    def _is_time_sensitive(self, query: str) -> bool:
        keywords = ("最新", "当前", "现在", "今年", "最近", "近期", "公告", "通知")
        return any(keyword in query for keyword in keywords) or bool(re.search(r"\b20\d{2}\b", query))

    def _is_active(self, doc: Document) -> bool:
        today = date.today().isoformat()
        effective_date = str(doc.metadata.get("effective_date", "")).strip()
        expire_date = str(doc.metadata.get("expire_date", "")).strip()
        if effective_date and effective_date > today:
            return False
        if expire_date and expire_date < today:
            return False
        return bool(effective_date or expire_date)

    def _is_expired(self, doc: Document) -> bool:
        expire_date = str(doc.metadata.get("expire_date", "")).strip()
        return bool(expire_date and expire_date < date.today().isoformat())

    def _query_intent_bonus(self, *, query: str, doc: Document) -> float:
        normalized_query = query.strip()
        if not normalized_query:
            return 0.0
        source_title = str(doc.metadata.get("source_title", ""))
        source_url = str(doc.metadata.get("source_url", ""))
        chunk_text = " ".join(
            [
                str(doc.metadata.get("chunk_text", "")),
                str(doc.metadata.get("section_hint", "")),
                doc.page_content,
            ]
        )
        bonus = 0.0
        college_name = self._extract_college_name(normalized_query)
        if college_name and college_name in source_title:
            bonus += 1.0
        if college_name and college_name in chunk_text:
            bonus += 0.6
        if "招生咨询电话" in normalized_query or "电话" in normalized_query:
            if any(token in chunk_text for token in ("联系方式：", "招生咨询电话", "咨询电话")):
                bonus += 1.4
            if "就业信息网" in source_title or "job.zut.edu.cn" in source_url:
                bonus -= 1.1
        if "网址" in normalized_query or "网站" in normalized_query:
            if any(token in chunk_text for token in ("网址：", "学院网址", "http://", "https://")):
                bonus += 1.4
            if "就业信息网" in source_title or "job.zut.edu.cn" in source_url:
                bonus -= 1.0
        if "国标代码" in normalized_query and "国标代码" in chunk_text:
            bonus += 1.2
        if "河南招生代码" in normalized_query and "河南招生代码" in chunk_text:
            bonus += 1.2
        if "调档比例" in normalized_query and "调档比例" in chunk_text:
            bonus += 1.0
        if "预留计划" in normalized_query and "预留计划" in chunk_text:
            bonus += 1.0
        return bonus

    def _extract_college_name(self, query: str) -> str:
        matched = re.search(r"([\u4e00-\u9fffA-Za-z（）()]{2,40}学院)", query)
        return matched.group(1) if matched else ""

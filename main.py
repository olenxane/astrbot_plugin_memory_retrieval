import json
import re
import uuid
import time
import base64
import difflib
from pathlib import Path

import numpy as np
import aiohttp

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import logger
from astrbot.core.agent.message import TextPart
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

try:
    from astrbot.api.web import PluginUploadFile, error_response, json_response, request
    HAS_ASTRBOT_WEB = True
except ImportError:
    from quart import jsonify, request as quart_request

    PluginUploadFile = None
    HAS_ASTRBOT_WEB = False

PLUGIN_NAME = "astrbot_plugin_memory_retrieval"
PLUGIN_DISPLAY_NAME = "memory_retrieval"
SUMMARY_NAME_RE = re.compile(r"^summary_[A-Za-z0-9_.-]+\.txt$")
RETRIEVAL_MODES = {"user", "llm", "both"}

def _json_ok(data):
    payload = {"status": "ok", "data": data}
    if HAS_ASTRBOT_WEB:
        return json_response(payload)
    return jsonify(payload)

def _json_err(msg, status_code=400):
    if HAS_ASTRBOT_WEB:
        return error_response(msg, status_code=status_code)
    return jsonify({"status": "error", "message": msg})

async def _request_json(default=None):
    if default is None:
        default = {}
    if HAS_ASTRBOT_WEB:
        return await request.json(default=default)
    try:
        payload = await quart_request.json
    except Exception:
        payload = default
    return default if payload is None else payload

async def _request_files():
    if HAS_ASTRBOT_WEB:
        return await request.files()
    return await quart_request.files

def _is_upload_file(upload):
    if upload is None:
        return False
    if HAS_ASTRBOT_WEB:
        return isinstance(upload, PluginUploadFile)
    return bool(getattr(upload, "filename", None)) and hasattr(upload, "read")

async def _read_upload_bytes(upload):
    raw = upload.read()
    if hasattr(raw, "__await__"):
        raw = await raw
    return raw

def _as_int(value, default, min_value=None, max_value=None):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value

def _as_float(value, default, min_value=None, max_value=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value

#文本分割
def split_text(text, chunk_size=500, overlap=50):
    if not text.strip():
        return []
    chunks = []
    paragraphs = text.split("\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = current + "\n" + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    seg = para[start:end].strip()
                    if seg:
                        chunks.append(seg)
                    start = end
                current = ""
            else:
                current = para
    if current.strip():
        chunks.append(current.strip())
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            overlapped.append(tail + " " + chunks[i])
        chunks = overlapped
    return [c for c in chunks if c.strip()]

class MemoryStore:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        self._load()

    def _load(self):
        index_file = self.data_dir / "index.json"
        if not index_file.exists():
            return
        with open(index_file, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        for doc_id in list(self.documents.keys()):
            emb_file = self.data_dir / "{}.npy".format(doc_id)
            if emb_file.exists():
                embeddings = np.load(str(emb_file))
                for i, chunk in enumerate(self.documents[doc_id]["chunks"]):
                    if i < len(embeddings):
                        chunk["embedding"] = embeddings[i].tolist()

    def _save(self):
        index_data = {}
        for doc_id, doc in self.documents.items():
            index_data[doc_id] = {
                "filename": doc["filename"],
                "content": doc.get("content", ""),
                "content_source": doc.get("content_source", ""),
                "chunks": [
                    {"text": c["text"], "chunk_id": c.get("chunk_id", "")}
                    for c in doc["chunks"]
                ],
                "created_at": doc.get("created_at", 0),
            }
        index_path = self.data_dir / "index.json"
        tmp_path = self.data_dir / "index.json.tmp"
        logger.info("[{}] 开始保存记忆索引，路径={}，临时文件={}，文档数={}，分块数={}".format(
            PLUGIN_DISPLAY_NAME,
            index_path,
            tmp_path,
            len(index_data),
            sum(len(doc.get("chunks", [])) for doc in self.documents.values()),
        ))
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(index_path)
        logger.info("[{}] 记忆索引保存完成，路径={}，是否存在={}，大小={}".format(
            PLUGIN_DISPLAY_NAME,
            index_path,
            index_path.exists(),
            index_path.stat().st_size if index_path.exists() else -1,
        ))
        for doc_id, doc in self.documents.items():
            vecs = [c["embedding"] for c in doc["chunks"] if "embedding" in c]
            if vecs:
                np.save(str(self.data_dir / "{}.npy".format(doc_id)), np.array(vecs, dtype=np.float32))

    def add_document(self, doc_id, filename, chunks_with_emb, content=""):
        self.documents[doc_id] = {
            "filename": filename,
            "content": content,
            "content_source": "upload" if content else "",
            "chunks": chunks_with_emb,
            "created_at": int(time.time()),
        }
        self._save()

    def update_document_chunks(self, doc_id, chunks_with_emb, content=None):
        if doc_id not in self.documents:
            return False
        if content is not None:
            self.documents[doc_id]["content"] = content
            if not self.documents[doc_id].get("content_source"):
                self.documents[doc_id]["content_source"] = "reconstructed"
        self.documents[doc_id]["chunks"] = chunks_with_emb
        self._save()
        return True

    def remove_document(self, doc_id):
        doc_id = str(doc_id)
        if doc_id not in self.documents:
            logger.warning("[{}] 删除记忆失败，文档不存在，doc_id={}，当前文档数={}".format(
                PLUGIN_DISPLAY_NAME,
                doc_id,
                len(self.documents),
            ))
            return False
        doc = self.documents[doc_id]
        emb = self.data_dir / "{}.npy".format(doc_id)
        logger.info("[{}] 开始删除记忆，doc_id={}，文件名={}，分块数={}，当前文档数={}，向量文件={}，是否存在={}".format(
            PLUGIN_DISPLAY_NAME,
            doc_id,
            doc.get("filename", ""),
            len(doc.get("chunks", [])),
            len(self.documents),
            emb,
            emb.exists(),
        ))
        removed = self.documents.pop(doc_id)
        try:
            self._save()
        except Exception:
            self.documents[doc_id] = removed
            logger.error("[{}] 删除记忆回滚，索引保存失败，doc_id={}".format(PLUGIN_DISPLAY_NAME, doc_id), exc_info=True)
            raise
        if emb.exists():
            try:
                emb.unlink()
                logger.info("[{}] 已删除向量文件，doc_id={}，路径={}".format(PLUGIN_DISPLAY_NAME, doc_id, emb))
            except Exception as e:
                logger.warning("[{}] 已删除索引，但删除向量文件失败，doc_id={}，错误={}".format(PLUGIN_DISPLAY_NAME, doc_id, e))
        removed_ok = doc_id not in self.documents
        logger.info("[{}] 删除记忆完成，doc_id={}，是否成功={}，剩余文档数={}".format(
            PLUGIN_DISPLAY_NAME,
            doc_id,
            removed_ok,
            len(self.documents),
        ))
        return removed_ok

    def search(self, query_embedding, top_k=3, threshold=0.5, doc_ids=None):
        """如传入 doc_ids，则只在这些文档内检索。目前该功能尚未启用"""
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        results = []
        for doc_id, doc in self.documents.items():
            if doc_ids is not None and doc_id not in doc_ids:
                continue
            for i, chunk in enumerate(doc["chunks"]):
                if "embedding" not in chunk:
                    continue
                c = np.array(chunk["embedding"], dtype=np.float32)
                c_norm = np.linalg.norm(c)
                if c_norm == 0:
                    continue
                sim = float(np.dot(q, c) / (q_norm * c_norm))
                if sim >= threshold:
                    results.append({
                        "doc_id": doc_id,
                        "filename": doc["filename"],
                        "chunk_id": chunk.get("chunk_id", "{}_{}".format(doc_id, i)),
                        "chunk_index": i,
                        "text": chunk["text"],
                        "score": round(sim, 4),
                    })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all_documents(self):
        return [
            {
                "doc_id": did,
                "filename": d["filename"],
                "chunk_count": len(d["chunks"]),
                "created_at": d.get("created_at", 0),
            }
            for did, d in self.documents.items()
        ]

    def get_total_chunks(self):
        return sum(len(d["chunks"]) for d in self.documents.values())

# 插件主体
class MemoryRetrievalPlugin(Star):
    def __init__(self, context, config):
        super().__init__(context)
        self.config = config

        data_path = Path(get_astrbot_data_path()) / PLUGIN_DISPLAY_NAME
        data_path.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path
        self.memory_store = MemoryStore(data_path / "vectors")

        # 总结目录
        self.summaries_dir = data_path / "summaries"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.summary_index_path = data_path / "summary_index.json"
        self.summary_index = self._load_summary_index()

        # 访问控制
        self.access_ctrl_path = data_path / "access_control.json"
        self.access_ctrl = self._load_access_ctrl()

        # 注册接口路由
        context.register_web_api("/{}/documents".format(PLUGIN_NAME), self.api_list_documents, ["GET"], "文档列表")
        context.register_web_api("/{}/upload".format(PLUGIN_NAME), self.api_upload_document, ["POST"], "上传文档")
        context.register_web_api("/{}/delete-document".format(PLUGIN_NAME), self.api_delete_document_post, ["POST"], "删除文档")
        context.register_web_api("/{}/documents/delete".format(PLUGIN_NAME), self.api_delete_document_post, ["POST"], "删除文档")
        context.register_web_api("/{}/documents/<doc_id>".format(PLUGIN_NAME), self.api_delete_document, ["DELETE", "POST"], "删除文档")
        context.register_web_api("/{}/config".format(PLUGIN_NAME), self.api_get_config, ["GET"], "读取配置")
        context.register_web_api("/{}/config/save".format(PLUGIN_NAME), self.api_save_config, ["POST"], "保存配置")
        context.register_web_api("/{}/search".format(PLUGIN_NAME), self.api_search, ["POST"], "检索记忆")
        context.register_web_api("/{}/stats".format(PLUGIN_NAME), self.api_stats, ["GET"], "统计信息")
        context.register_web_api("/{}/clear-session".format(PLUGIN_NAME), self.api_clear_session, ["POST"], "清理会话缓存")
        context.register_web_api("/{}/test-embedding".format(PLUGIN_NAME), self.api_test_embedding, ["POST"], "测试嵌入接口")
        context.register_web_api("/{}/access-control".format(PLUGIN_NAME), self.api_get_access_control, ["GET"], "读取访问控制")
        context.register_web_api("/{}/access-control/save".format(PLUGIN_NAME), self.api_save_access_control, ["POST"], "保存访问控制")
        context.register_web_api("/{}/summaries".format(PLUGIN_NAME), self.api_list_summaries, ["GET"], "总结列表")
        context.register_web_api("/{}/summaries/<filename>".format(PLUGIN_NAME), self.api_get_summary, ["GET"], "读取总结")
        context.register_web_api("/{}/summaries/<filename>/save".format(PLUGIN_NAME), self.api_save_summary, ["POST"], "保存总结")
        context.register_web_api("/{}/summaries/<filename>/delete".format(PLUGIN_NAME), self.api_delete_summary, ["POST"], "删除总结")

        logger.info("[{}] 插件已加载，文档数：{}，分块数：{}".format(
            PLUGIN_DISPLAY_NAME, len(self.memory_store.documents), self.memory_store.get_total_chunks()))

    #  访问控制

    def _load_access_ctrl(self):
        default = {
            "access_mode": "disabled",
            "whitelist": [],
            "blacklist": [],
            "user_docs": {},
        }
        if self.access_ctrl_path.exists():
            try:
                with open(self.access_ctrl_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k in default:
                        if k not in data:
                            data[k] = default[k]
                    return data
            except Exception:
                pass
        return default

    def _save_access_ctrl(self):
        with open(self.access_ctrl_path, "w", encoding="utf-8") as f:
            json.dump(self.access_ctrl, f, ensure_ascii=False, indent=2)

    # 总结索引
    def _load_summary_index(self):
        default = {"sessions": {}, "files": {}}
        if not self.summary_index_path.exists():
            return default
        try:
            with open(self.summary_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return default
            sessions = data.get("sessions")
            files = data.get("files")
            if not isinstance(sessions, dict):
                sessions = {}
            if not isinstance(files, dict):
                files = {}
            return {"sessions": sessions, "files": files}
        except Exception as e:
            logger.warning("[{}] 读取总结索引失败，将使用空索引：{}".format(PLUGIN_DISPLAY_NAME, e))
            return default

    def _save_summary_index(self):
        self.summary_index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.summary_index_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_index, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.summary_index_path)

    def _summary_session_key(self, unified_msg_origin, conversation_id):
        return "{}::{}".format(str(unified_msg_origin), str(conversation_id))

    def _find_summary_doc_id(self, filename):
        filename = Path(str(filename)).name
        record = self.summary_index.get("files", {}).get(filename)
        if isinstance(record, dict):
            doc_id = str(record.get("doc_id") or "").strip()
            return doc_id or None
        for record in self.summary_index.get("sessions", {}).values():
            if isinstance(record, dict) and Path(str(record.get("filename") or "")).name == filename:
                doc_id = str(record.get("doc_id") or "").strip()
                return doc_id or None
        return None

    def _set_summary_record(self, filename, doc_id, session_key=None):
        filename = Path(str(filename)).name
        doc_id = str(doc_id or "").strip()
        record = {
            "filename": filename,
            "doc_id": doc_id,
            "updated_at": int(time.time()),
        }
        self.summary_index.setdefault("files", {})[filename] = record
        if session_key:
            self.summary_index.setdefault("sessions", {})[str(session_key)] = record
        self._save_summary_index()

    def _remove_summary_record_by_filename(self, filename):
        filename = Path(str(filename)).name
        changed = False
        files = self.summary_index.setdefault("files", {})
        if filename in files:
            files.pop(filename, None)
            changed = True
        sessions = self.summary_index.setdefault("sessions", {})
        for key, record in list(sessions.items()):
            if isinstance(record, dict) and Path(str(record.get("filename") or "")).name == filename:
                sessions.pop(key, None)
                changed = True
        if changed:
            self._save_summary_index()
        return changed

    async def _add_summary_to_store(self, filename, content, doc_id=None):
        filename = Path(str(filename)).name
        content = str(content or "")
        doc_id = str(doc_id or "").strip() or self._find_summary_doc_id(filename)
        if not doc_id:
            doc_id = uuid.uuid4().hex[:8]
        try:
            chunk_size = _as_int(self.config.get("chunk_size", 500), 500, 100, 2000)
            chunk_overlap = _as_int(self.config.get("chunk_overlap", 50), 50, 0, min(500, chunk_size - 1))
            chunks_data = await self._build_chunks_with_embeddings(doc_id, content, chunk_size, chunk_overlap)
            updated = False
            if doc_id in self.memory_store.documents:
                updated = self.memory_store.update_document_chunks(doc_id, chunks_data, content=content)
                if updated:
                    self.memory_store.documents[doc_id]["filename"] = filename
                    self.memory_store._save()
            if not updated:
                self.memory_store.add_document(doc_id, filename, chunks_data, content=content)
            return {"ok": True, "doc_id": doc_id, "updated": updated}
        except Exception as e:
            logger.error("[{}] 同步总结到历史记忆失败：{}".format(PLUGIN_DISPLAY_NAME, e), exc_info=True)
            return {"ok": False, "error": str(e)}

    def _check_access(self, event):
        """返回用户是否有权使用历史记忆。"""
        mode = self.access_ctrl.get("access_mode", "disabled")
        if mode == "disabled":
            return True
        sender_id = str(event.get_sender_id())
        if mode == "whitelist":
            return sender_id in [str(x) for x in self.access_ctrl.get("whitelist", [])]
        if mode == "blacklist":
            return sender_id not in [str(x) for x in self.access_ctrl.get("blacklist", [])]
        return True

    def _get_user_doc_ids(self, event):
        """返回当前用户可访问的文档 ID 列表，None 表示可访问全部文档。"""
        sender_id = str(event.get_sender_id())
        user_docs = self.access_ctrl.get("user_docs", {})
        if sender_id in user_docs and user_docs[sender_id]:
            return user_docs[sender_id]
        return None

    def _get_retrieval_mode(self):
        mode = str(self.config.get("retrieval_mode", "user")).strip().lower()
        return mode if mode in RETRIEVAL_MODES else "user"

    def _retrieval_enabled_for(self, source):
        mode = self._get_retrieval_mode()
        if source == "user":
            return mode in ("user", "both")
        if source == "llm":
            return mode in ("llm", "both")
        return False

    #  嵌入接口

    def _get_api_config(self):
        api_base = self.config.get("embedding_api_base", "").strip()
        api_key = self.config.get("embedding_api_key", "").strip()
        model = self.config.get("embedding_model", "").strip()
        timeout = _as_int(self.config.get("timeout", 60), 60, 10, 300)
        if not api_base:
            raise ValueError("请先在插件配置中填写 嵌入接口 Base URL")
        if not model:
            raise ValueError("请先在插件配置中填写 Embedding 模型名称")
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = api_base + "/v1"
        return api_base, api_key, model, timeout

    async def _get_embeddings(self, texts):
        api_base, api_key, model, timeout = self._get_api_config()
        url = "{}/embeddings".format(api_base)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer {}".format(api_key)
        all_embeddings = []
        batch_size = 16
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                payload = {"input": batch, "model": model, "encoding_format": "float"}
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        raise Exception("嵌入接口 {}: {}".format(resp.status, err[:200]))
                    data = await resp.json()
                    items = data.get("data", [])
                    items.sort(key=lambda x: x.get("index", 0))
                    for item in items:
                        all_embeddings.append(item["embedding"])
        return all_embeddings

    async def _chat_llm(self, umo, prompt):
        """使用自定义 OpenAI 兼容接口生成对话总结。"""
        api_base = self.config.get("summary_api_base", "").strip()
        api_key = self.config.get("summary_api_key", "").strip()
        model = self.config.get("summary_model", "").strip()
        timeout = _as_int(self.config.get("timeout", 60), 60, 10, 300)
        if not api_base:
            raise ValueError("请先配置总结模型 API Base URL")
        if not model:
            raise ValueError("请先配置总结模型名称")
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = api_base + "/v1"
        url = "{}/chat/completions".format(api_base)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer {}".format(api_key)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    raise Exception("总结模型 API 返回错误 {}: {}".format(resp.status, err[:200]))
                data = await resp.json()
                content = self._extract_chat_content(data)
                if content:
                    return content
                preview = json.dumps(data, ensure_ascii=False)[:1000]
                logger.warning("[{}] 总结模型响应中没有可用正文，响应摘要：{}".format(PLUGIN_DISPLAY_NAME, preview))
                raise Exception("总结模型没有返回有效正文，请检查模型名称、接口类型或响应格式")

    def _extract_chat_content(self, data):
        if not isinstance(data, dict):
            return ""
        direct = data.get("output_text") or data.get("text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        choices = data.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                text = choice.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
                message = choice.get("message") or choice.get("delta") or {}
                if not isinstance(message, dict):
                    continue
                content = self._normalize_chat_content(message.get("content"))
                if content:
                    return content
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
        message = data.get("message")
        if isinstance(message, dict):
            content = self._normalize_chat_content(message.get("content"))
            if content:
                return content
        return ""

    def _normalize_chat_content(self, content):
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
        return ""

    def _doc_source_text(self, doc):
        content = str(doc.get("content", "") or "")
        if doc.get("content_source") == "upload" and content.strip():
            return content
        reconstructed = self._merge_overlapped_chunks(doc.get("chunks", []))
        return reconstructed if reconstructed.strip() else content

    def _merge_overlapped_chunks(self, chunks):
        texts = [str(c.get("text", "")) for c in chunks if str(c.get("text", "")).strip()]
        if not texts:
            return ""
        merged = texts[0]
        for text in texts[1:]:
            best = 0
            max_len = min(len(merged), len(text), 2000)
            for n in range(max_len, 0, -1):
                if merged[-n:] == text[:n]:
                    best = n
                    break
            if best:
                merged += text[best:]
            else:
                merged += "\n\n" + text
        return merged

    def _canonical_rebuild_sources(self):
        sources = {}
        for doc_id, doc in self.memory_store.documents.items():
            sources[doc_id] = self._doc_source_text(doc)

        by_filename = {}
        for doc_id, doc in self.memory_store.documents.items():
            filename = str(doc.get("filename", ""))
            by_filename.setdefault(filename, []).append(doc_id)

        for doc_ids in by_filename.values():
            if len(doc_ids) < 2:
                continue
            candidates = [(doc_id, sources.get(doc_id, "")) for doc_id in doc_ids if sources.get(doc_id, "").strip()]
            if len(candidates) < 2:
                continue
            # 同名文件重复上传后，旧分块可能带有重叠内容。
            # 这里优先选更短且高度相似的原文，避免后续重建越滚越长。
            shortest_id, shortest_text = min(candidates, key=lambda item: len(item[1]))
            for doc_id, text in candidates:
                if doc_id == shortest_id:
                    continue
                shorter = min(len(shortest_text), len(text))
                longer = max(len(shortest_text), len(text))
                similar = difflib.SequenceMatcher(None, shortest_text, text).ratio()
                if shorter > 0 and shorter / longer >= 0.75 and similar >= 0.8:
                    sources[doc_id] = shortest_text
        return sources

    async def _build_chunks_with_embeddings(self, doc_id, content, chunk_size, chunk_overlap):
        text_chunks = split_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
        if not text_chunks:
            raise ValueError("没有可用分块")
        embeddings = await self._get_embeddings(text_chunks)
        if len(embeddings) != len(text_chunks):
            raise ValueError("分块数与嵌入数不一致，期望 {}，实际 {}".format(len(text_chunks), len(embeddings)))
        chunks_data = []
        for i, (text, emb) in enumerate(zip(text_chunks, embeddings)):
            chunks_data.append({"text": text, "embedding": emb, "chunk_id": "{}_{}".format(doc_id, i)})
        return chunks_data

    async def _search_memory(self, query, event=None, top_k=None, threshold=None):
        query = str(query or "").strip()
        if not query:
            return []
        if event is not None and not self._check_access(event):
            return []
        if not self.memory_store.documents:
            return []

        q_embs = await self._get_embeddings([query])
        if not q_embs:
            return []
        if top_k is None:
            top_k = self.config.get("top_k", 3)
        if threshold is None:
            threshold = self.config.get("similarity_threshold", 0.5)
        top_k = _as_int(top_k, 3, 1, 20)
        threshold = _as_float(threshold, 0.5, 0.0, 1.0)
        doc_ids = self._get_user_doc_ids(event) if event is not None else None
        return self.memory_store.search(q_embs[0], top_k=top_k, threshold=threshold, doc_ids=doc_ids)

    async def _rebuild_all_documents(self, chunk_size, chunk_overlap):
        rebuilt = 0
        sources = self._canonical_rebuild_sources()
        for doc_id, doc in list(self.memory_store.documents.items()):
            content = sources.get(doc_id, "")
            if not content.strip():
                continue
            chunks_data = await self._build_chunks_with_embeddings(doc_id, content, chunk_size, chunk_overlap)
            self.memory_store.update_document_chunks(doc_id, chunks_data, content=content)
            rebuilt += 1
        return rebuilt

    # 对话总结命令

    @filter.command("summem")
    async def summem_handler(self, event: AstrMessageEvent):
        """总结当前会话，保存为 txt，并同步导入历史记忆。"""
        yield event.plain_result("正在总结当前会话，请稍候...")
        try:
            umo = event.unified_msg_origin
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if not curr_cid:
                yield event.plain_result("未找到当前会话，无法生成总结。")
                return

            conversation = await conv_mgr.get_conversation(umo, curr_cid)
            if not conversation:
                yield event.plain_result("读取当前会话失败，无法生成总结。")
                return

            history_str = conversation.history or ""
            if not history_str.strip():
                yield event.plain_result("当前会话没有可总结的历史记录。")
                return

            summary_prompt = (
                "你是对话摘要助手。请保留事实、人物/实体、约束与未完成事项，避免输出 JSON 或代码块。\n"
                "【核心任务】以 AI 第一人称记录所有对话内容。\n"
                "\n"
                "【必须完整保留的8大要素】\n"
                "1. 时间信息：具体时刻、时间段、持续时间。\n"
                "2. 参与者身份：姓名、角色、特征。\n"
                "3. 核心事件：完整经过、重要决定、承诺、约定。\n"
                "4. 情感细节：具体表现，避免只写笼统情绪。\n"
                "5. 关键对话：表白、重要比喻、核心隐喻，尽量保留原话。\n"
                "6. 物品与场景：名称、规格、颜色、地点。\n"
                "7. 量化数据：数量、尺寸、次数。\n"
                "8. 特殊需求与分歧：需求、争议、解决方案。\n"
                "\n"
                "【完整性标准】\n"
                "必须完整除了系统提示以外的记录所有内容，不遗漏任何重要信息。每个事件都要记录：谁、何时、在哪、做了什么、说了什么、有什么反应。宁可详细也不删减。\n"
                "如果出现以\"[来源：xxxx.txt]\"开头的类似历史记忆的内容，则该段内容不参与总结。\n"
                "\n"
                "【输出格式】\n"
                "流畅的多段叙述，段内按时间顺序连贯表述，不用序号、不分点。\n"
                "\n"
                "【严格禁止】\n"
                "不总结系统提示词、不输出 JSON/代码块、不添加“总之”等总结词、不用“早期/中期/近期”等模糊划分、不使用序号分点。\n"
                "\n"
                "【输出语言】简体中文。\n"
                "\n"
                "对话历史：\n"
                "{}"
            ).format(history_str[-6000:])
            summary = await self._chat_llm(umo, summary_prompt)
            ts = int(time.time())
            session_key = self._summary_session_key(umo, curr_cid)
            existing = self.summary_index.get("sessions", {}).get(session_key)
            target_doc_id = None
            is_update = False

            if isinstance(existing, dict):
                old_filename = Path(str(existing.get("filename") or "")).name
                old_path = self._summary_path(old_filename)
                if old_path is not None and old_path.exists():
                    filename = old_filename
                    filepath = old_path
                    target_doc_id = str(existing.get("doc_id") or "") or None
                    is_update = True
                elif old_filename:
                    self._remove_summary_record_by_filename(old_filename)

            if not is_update:
                safe_umo = re.sub(r"[^A-Za-z0-9_.-]", "_", umo)[:40]
                safe_cid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(curr_cid))[:24]
                filename = "summary_{}_{}_{}.txt".format(ts, safe_umo, safe_cid)
                filepath = self.summaries_dir / filename

            header = "# 对话总结\n# 会话来源：{}\n# 会话 ID：{}\n# 生成时间：{}\n# 时间戳：{}\n\n".format(
                umo, curr_cid, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), ts
            )
            full_content = header + summary
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)

            store_result = await self._add_summary_to_store(filename, full_content, doc_id=target_doc_id)
            if not store_result.get("ok"):
                msg = "对话总结已保存为 {}，但导入历史记忆失败：{}".format(
                    filename, store_result.get("error") or "未知错误")
                logger.error("[{}] {}".format(PLUGIN_DISPLAY_NAME, msg))
                yield event.plain_result(msg)
                return

            self._set_summary_record(filename, store_result.get("doc_id"), session_key=session_key)
            action = "已覆盖本会话旧总结" if is_update or store_result.get("updated") else "已生成新总结"
            yield event.plain_result("对话总结完成，{}并已导入历史记忆：{}\n\n{}".format(action, filename, summary))

        except Exception as e:
            msg = "生成对话总结失败：{}".format(e)
            logger.error("[{}] {}".format(PLUGIN_DISPLAY_NAME, msg), exc_info=True)
            yield event.plain_result(msg)

    #  消息注入

    async def _get_session_key(self, event):
        """按消息来源和当前会话 ID 隔离已注入片段。"""
        umo = event.unified_msg_origin
        cid = None
        try:
            conv_mgr = self.context.conversation_manager
            cid = await conv_mgr.get_curr_conversation_id(umo)
        except Exception as e:
            logger.warning("[{}] 获取当前会话 ID 失败，回退到仅使用消息来源作为键：{}".format(PLUGIN_DISPLAY_NAME, e))
        if cid:
            return "sent_chunks_{}_{}".format(umo, cid), umo, cid
        return "sent_chunks_{}".format(umo), umo, None

    @filter.on_llm_request()
    async def on_llm_request(self, event, req):
        try:
            if not self._retrieval_enabled_for("user"):
                return
            query = event.message_str.strip()
            if not query:
                return
            try:
                results = await self._search_memory(query, event=event)
            except Exception as e:
                logger.warning("[{}] 生成查询嵌入失败：{}".format(PLUGIN_DISPLAY_NAME, e))
                return
            if not results:
                return

            session_key, umo, cid = await self._get_session_key(event)
            sent_ids = await self.get_kv_data(session_key, [])
            if not isinstance(sent_ids, list):
                sent_ids = []
            sent_set = set(sent_ids)
            new_results = [r for r in results if r["chunk_id"] not in sent_set]
            if not new_results:
                return

            memory_parts = ["[来源：{}]\n{}".format(r["filename"], r["text"]) for r in new_results]
            memory_context = "\n\n".join(memory_parts)
            req.extra_user_content_parts.append(
                TextPart(text=(
                    "<memory_retrieval_context>\n"
                    "以下内容来自 memory_retrieval 历史记忆，可作为回答参考：\n\n"
                    "{}\n"
                    "</memory_retrieval_context>".format(memory_context)
                ))
            )
            sent_ids.extend([r["chunk_id"] for r in new_results])
            await self.put_kv_data(session_key, sent_ids)
            logger.info("[{}] 已为会话注入 {} 个历史记忆片段，umo={}，cid={}".format(PLUGIN_DISPLAY_NAME, len(new_results), umo[:40], cid))
        except Exception as e:
            logger.error("[{}] LLM 请求注入处理失败：{}".format(PLUGIN_DISPLAY_NAME, e), exc_info=True)

    @filter.llm_tool(name="memory_search")
    async def memory_search(self, event: AstrMessageEvent, query: str, top_k: int = 0):
        """当用户的消息中出现一些特殊名词或事件或记忆，如“陈铮暄”（人名）、“我昨晚干了什么”（事件）、“我的生日是？”（记忆）时，可以使用该工具检索记忆库以获得相关内容的记忆。

        Args:
            query(string): 使用关键词或句子检索记忆，会进行一次向量检索。
            top_k(number): 返回结果数，范围 1-20。传 0 或省略时使用插件配置，推荐省略以使用默认配置。
        """
        if not self._retrieval_enabled_for("llm"):
            return "[记忆检索] LLM 工具检索未启用，请将 retrieval_mode 设置为 llm 或 both。"
        query = str(query or "").strip()
        if not query:
            return "[记忆检索] 缺少 query 参数，请提供要检索的问题。"
        try:
            configured_top_k = self.config.get("top_k", 3)
            top_k = configured_top_k if _as_int(top_k, 0, 0, 20) == 0 else top_k
            top_k = _as_int(top_k, configured_top_k, 1, 20)
            results = await self._search_memory(query, event=event, top_k=top_k)
        except Exception as e:
            logger.error("[{}] memory_search 工具调用失败：{}".format(PLUGIN_DISPLAY_NAME, e), exc_info=True)
            return "[记忆检索] 历史记忆检索失败：{}".format(e)
        if not results:
            return "没有检索到相关历史记忆。"
        lines = []
        for i, item in enumerate(results, 1):
            lines.append(
                "{}. 来源：{}\n相似度：{}\n内容：\n{}".format(
                    i,
                    item.get("filename", ""),
                    item.get("score", ""),
                    item.get("text", ""),
                )
            )
        return "\n\n---\n\n".join(lines)

    async def api_list_documents(self):
        return _json_ok({"documents": self.memory_store.get_all_documents()})

    async def api_stats(self):
        docs = self.memory_store.get_all_documents()
        return _json_ok({"total_documents": len(docs), "total_chunks": self.memory_store.get_total_chunks(), "documents": docs})

    async def api_upload_document(self):
        payload = await _request_json(default={})
        raw = b""
        fname = "unknown.txt"

        if isinstance(payload, dict) and payload.get("content_base64"):
            fname = Path(str(payload.get("filename") or "unknown.txt")).name
            try:
                raw = base64.b64decode(str(payload.get("content_base64")), validate=True)
            except Exception:
                return _json_err("上传内容无效", 400)
        else:
            files = await _request_files()
            upload = files.get("file")
            if not _is_upload_file(upload):
                return _json_err("缺少上传文件", 400)
            fname = Path(upload.filename or "unknown.txt").name
            raw = await _read_upload_bytes(upload)

        if not fname.lower().endswith(".txt"):
            return _json_err("仅支持 .txt 文件", 400)
        content = ""
        for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
            try:
                content = raw.decode(enc) if isinstance(raw, bytes) else raw
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if not content.strip():
            return _json_err("文件内容为空", 400)
        chunk_size = _as_int(self.config.get("chunk_size", 500), 500, 100, 2000)
        chunk_overlap = _as_int(self.config.get("chunk_overlap", 50), 50, 0, min(500, chunk_size - 1))
        doc_id = uuid.uuid4().hex[:8]
        try:
            chunks_data = await self._build_chunks_with_embeddings(doc_id, content, chunk_size, chunk_overlap)
        except Exception as e:
            return _json_err("生成嵌入失败：{}".format(e), 500)
        self.memory_store.add_document(doc_id, fname, chunks_data, content=content)
        return _json_ok({"doc_id": doc_id, "filename": fname, "chunk_count": len(chunks_data)})

    async def api_delete_document(self, doc_id):
        doc_id = str(doc_id).strip()
        available_ids = list(self.memory_store.documents.keys())
        logger.info("[{}] 开始处理删除请求，doc_id={}，是否存在={}，当前文档数={}，示例文档={}".format(
            PLUGIN_DISPLAY_NAME,
            doc_id,
            doc_id in self.memory_store.documents,
            len(available_ids),
            available_ids[:50],
        ))
        if doc_id not in self.memory_store.documents:
            logger.warning("[{}] 删除失败，文档不存在，doc_id={}，示例文档={}".format(PLUGIN_DISPLAY_NAME, doc_id, available_ids[:50]))
            return _json_err("未找到文档", 404)
        doc = self.memory_store.documents[doc_id]
        logger.info("[{}] 删除目标已确认，doc_id={}，文件名={}，分块数={}，创建时间={}".format(
            PLUGIN_DISPLAY_NAME,
            doc_id,
            doc.get("filename", ""),
            len(doc.get("chunks", [])),
            doc.get("created_at", 0),
        ))
        try:
            removed = self.memory_store.remove_document(doc_id)
            self._remove_deleted_doc_assignments(doc_id)
        except Exception as e:
            logger.error("[{}] 删除文档失败，doc_id={}，错误={}".format(PLUGIN_DISPLAY_NAME, doc_id, e), exc_info=True)
            return _json_err("删除文档失败：{}".format(e), 500)
        if not removed or doc_id in self.memory_store.documents:
            logger.error("[{}] 删除后状态异常，doc_id={}，removed={}，still_exists={}".format(
                PLUGIN_DISPLAY_NAME,
                doc_id,
                removed,
                doc_id in self.memory_store.documents,
            ))
            return _json_err("删除失败：索引未正确更新", 500)
        docs = self.memory_store.get_all_documents()
        logger.info("[{}] 文档删除完成，doc_id={}，剩余文档数={}".format(PLUGIN_DISPLAY_NAME, doc_id, len(docs)))
        return _json_ok({
            "deleted": doc_id,
            "documents": docs,
            "total_documents": len(docs),
            "total_chunks": self.memory_store.get_total_chunks(),
        })

    async def api_delete_document_post(self):
        payload = await _request_json(default={})
        logger.info("[{}] 收到删除请求，载荷类型={}，字段={}".format(
            PLUGIN_DISPLAY_NAME,
            type(payload).__name__,
            list(payload.keys()) if isinstance(payload, dict) else [],
        ))
        if not isinstance(payload, dict):
            logger.warning("[{}] 删除请求格式无效，载荷类型={}".format(PLUGIN_DISPLAY_NAME, type(payload).__name__))
            return _json_err("请求体不是合法 JSON 对象", 400)
        doc_id = str(payload.get("doc_id") or payload.get("id") or "").strip()
        logger.info("[{}] 删除请求解析完成，doc_id={}，当前文档数={}".format(
            PLUGIN_DISPLAY_NAME,
            doc_id,
            len(self.memory_store.documents),
        ))
        if not doc_id:
            logger.warning("[{}] 删除请求缺少 doc_id，字段={}".format(PLUGIN_DISPLAY_NAME, list(payload.keys())))
            return _json_err("缺少 doc_id", 400)
        return await self.api_delete_document(doc_id)

    def _remove_deleted_doc_assignments(self, doc_id):
        changed = False
        affected_users = []
        user_docs = self.access_ctrl.get("user_docs", {})
        if not isinstance(user_docs, dict):
            logger.warning("[{}] 跳过用户文档清理，user_docs 不是字典，doc_id={}".format(PLUGIN_DISPLAY_NAME, doc_id))
            return
        for uid in list(user_docs.keys()):
            old_docs = user_docs.get(uid, [])
            if not isinstance(old_docs, list):
                del user_docs[uid]
                changed = True
                affected_users.append(str(uid))
                continue
            new_docs = [item for item in old_docs if str(item) != str(doc_id)]
            if len(new_docs) != len(old_docs):
                changed = True
                affected_users.append(str(uid))
                if new_docs:
                    user_docs[uid] = new_docs
                else:
                    del user_docs[uid]
        if changed:
            self._save_access_ctrl()
        logger.info("[{}] 用户文档清理完成，doc_id={}，是否变更={}，影响用户={}".format(
            PLUGIN_DISPLAY_NAME,
            doc_id,
            changed,
            affected_users[:50],
        ))

    async def api_get_config(self):
        return _json_ok({
            "embedding_api_base": self.config.get("embedding_api_base", ""),
            "embedding_api_key": self.config.get("embedding_api_key", ""),
            "embedding_model": self.config.get("embedding_model", ""),
            "summary_api_base": self.config.get("summary_api_base", ""),
            "summary_api_key": self.config.get("summary_api_key", ""),
            "summary_model": self.config.get("summary_model", ""),
            "chunk_size": self.config.get("chunk_size", 500),
            "chunk_overlap": self.config.get("chunk_overlap", 50),
            "top_k": self.config.get("top_k", 3),
            "similarity_threshold": self.config.get("similarity_threshold", 0.5),
            "retrieval_mode": self._get_retrieval_mode(),
            "timeout": self.config.get("timeout", 60),
        })

    async def api_save_config(self):
        payload = await _request_json(default={})
        if not isinstance(payload, dict):
            return _json_err("请求体不是合法 JSON 对象", 400)

        old_chunk_size = _as_int(self.config.get("chunk_size", 500), 500, 100, 2000)
        old_chunk_overlap = _as_int(self.config.get("chunk_overlap", 50), 50, 0, min(500, old_chunk_size - 1))

        for key in ("embedding_api_base", "embedding_api_key", "embedding_model", "summary_api_base", "summary_api_key", "summary_model"):
            if key in payload:
                self.config[key] = str(payload.get(key, "")).strip()

        if "chunk_size" in payload:
            self.config["chunk_size"] = _as_int(payload["chunk_size"], 500, 100, 2000)
        if "chunk_overlap" in payload:
            max_overlap = max(0, _as_int(self.config.get("chunk_size", 500), 500, 100, 2000) - 1)
            self.config["chunk_overlap"] = _as_int(payload["chunk_overlap"], 50, 0, min(500, max_overlap))
        if "top_k" in payload:
            self.config["top_k"] = _as_int(payload["top_k"], 3, 1, 20)
        if "similarity_threshold" in payload:
            self.config["similarity_threshold"] = _as_float(payload["similarity_threshold"], 0.5, 0.0, 1.0)
        if "retrieval_mode" in payload:
            mode = str(payload.get("retrieval_mode", "user")).strip().lower()
            self.config["retrieval_mode"] = mode if mode in RETRIEVAL_MODES else "user"
        if "timeout" in payload:
            self.config["timeout"] = _as_int(payload["timeout"], 60, 10, 300)

        new_chunk_size = _as_int(self.config.get("chunk_size", 500), 500, 100, 2000)
        new_chunk_overlap = _as_int(self.config.get("chunk_overlap", 50), 50, 0, min(500, new_chunk_size - 1))
        rebuilt = 0
        force_rebuild = bool(payload.get("force_rebuild"))
        if (force_rebuild or (new_chunk_size, new_chunk_overlap) != (old_chunk_size, old_chunk_overlap)) and self.memory_store.documents:
            try:
                rebuilt = await self._rebuild_all_documents(new_chunk_size, new_chunk_overlap)
            except Exception as e:
                self.config["chunk_size"] = old_chunk_size
                self.config["chunk_overlap"] = old_chunk_overlap
                return _json_err("重建分块失败，配置未保存：{}".format(e), 500)

        self.config.save_config()
        return _json_ok({"saved": True, "rebuilt_documents": rebuilt})

    async def api_search(self):
        payload = await _request_json(default={})
        if not isinstance(payload, dict):
            payload = {}
        query = str(payload.get("query", "")).strip()
        if not query:
            return _json_err("缺少检索内容", 400)
        try:
            results = await self._search_memory(query)
            return _json_ok({"results": results})
        except Exception as e:
            return _json_err("检索失败：{}".format(e), 500)

    async def api_clear_session(self):
        payload = await _request_json(default={})
        if not isinstance(payload, dict):
            payload = {}
        umo = str(payload.get("unified_msg_origin", "") or payload.get("umo", "") or "").strip()
        cid = str(payload.get("conversation_id", "") or payload.get("cid", "") or "").strip()
        raw_key = bool(payload.get("raw_key"))
        session_id = str(payload.get("session_id", "") or "").strip()

        if umo and cid:
            key = "sent_chunks_{}_{}".format(umo, cid)
            await self.delete_kv_data(key)
            return _json_ok({"cleared": key, "unified_msg_origin": umo, "conversation_id": cid})
        if session_id:
            key = session_id if raw_key else "sent_chunks_{}".format(session_id)
            await self.delete_kv_data(key)
            return _json_ok({"cleared": key})
        return _json_err("需要 unified_msg_origin 和 conversation_id，或传入 session_id", 400)

    async def api_test_embedding(self):
        try:
            q_embs = await self._get_embeddings(["hello world test"])
            if not q_embs:
                return _json_err("嵌入接口返回为空", 500)
            dim = len(q_embs[0])
            return _json_ok({"ok": True, "dimension": dim, "message": "嵌入接口连接成功"})
        except Exception as e:
            return _json_err("测试失败：{}".format(e), 500)

    # 访问控制接口

    async def api_get_access_control(self):
        return _json_ok({
            "access_mode": self.access_ctrl.get("access_mode", "disabled"),
            "whitelist": self.access_ctrl.get("whitelist", []),
            "blacklist": self.access_ctrl.get("blacklist", []),
            "user_docs": self.access_ctrl.get("user_docs", {}),
            "available_docs": self.memory_store.get_all_documents(),
        })

    async def api_save_access_control(self):
        payload = await _request_json(default={})
        if not isinstance(payload, dict):
            return _json_err("请求体不是合法 JSON 对象", 400)
        if "access_mode" in payload:
            mode = str(payload["access_mode"])
            if mode not in ("disabled", "whitelist", "blacklist"):
                return _json_err("访问模式无效", 400)
            self.access_ctrl["access_mode"] = mode
        if "whitelist" in payload:
            self.access_ctrl["whitelist"] = self._normalize_id_list(payload["whitelist"])
        if "blacklist" in payload:
            self.access_ctrl["blacklist"] = self._normalize_id_list(payload["blacklist"])
        if "user_docs" in payload:
            self.access_ctrl["user_docs"] = self._normalize_user_docs(payload["user_docs"])
        self._save_access_ctrl()
        return _json_ok({"saved": True})

    def _normalize_id_list(self, value):
        if not isinstance(value, list):
            return []
        result = []
        seen = set()
        for item in value:
            uid = str(item).strip()
            if uid and uid not in seen:
                seen.add(uid)
                result.append(uid)
        return result

    def _normalize_user_docs(self, value):
        if not isinstance(value, dict):
            return {}
        valid_docs = set(self.memory_store.documents.keys())
        result = {}
        for uid, doc_ids in value.items():
            uid = str(uid).strip()
            if not uid or not isinstance(doc_ids, list):
                continue
            clean_doc_ids = []
            seen = set()
            for doc_id in doc_ids:
                doc_id = str(doc_id).strip()
                if doc_id in valid_docs and doc_id not in seen:
                    seen.add(doc_id)
                    clean_doc_ids.append(doc_id)
            if clean_doc_ids:
                result[uid] = clean_doc_ids
        return result

    #  总结接口

    async def api_list_summaries(self):
        files = []
        for f in sorted(self.summaries_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True):
            # 读取前几行作为预览
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
                preview = ""
                content_started = False
                for line in lines:
                    if line.startswith("#"):
                        continue
                    content_started = True
                    preview += line
                    if len(preview) > 200:
                        break
                files.append({
                    "filename": f.name,
                    "size": f.stat().st_size,
                    "modified": int(f.stat().st_mtime),
                    "preview": preview.strip()[:200],
                })
            except Exception:
                files.append({"filename": f.name, "size": 0, "modified": 0, "preview": ""})
        return _json_ok({"summaries": files})

    async def api_get_summary(self, filename):
        filepath = self._summary_path(filename)
        if filepath is None or not filepath.exists() or not filepath.is_file():
            return _json_err("未找到总结文件", 404)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return _json_ok({"filename": filename, "content": content})
        except Exception as e:
            return _json_err("读取总结失败：{}".format(e), 500)

    async def api_save_summary(self, filename):
        filepath = self._summary_path(filename)
        if filepath is None or not filepath.exists() or not filepath.is_file():
            return _json_err("未找到总结文件", 404)
        payload = await _request_json(default={})
        if not isinstance(payload, dict):
            payload = {}
        content = payload.get("content", "")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            doc_id = self._find_summary_doc_id(filename)
            store_result = await self._add_summary_to_store(filename, content, doc_id=doc_id)
            if not store_result.get("ok"):
                return _json_err("总结已保存，但同步历史记忆失败：{}".format(store_result.get("error") or "未知错误"), 500)
            self._set_summary_record(filename, store_result.get("doc_id"))
            return _json_ok({
                "saved": True,
                "filename": filename,
                "memory_doc_id": store_result.get("doc_id"),
                "memory_synced": True,
            })
        except Exception as e:
            return _json_err("写入总结失败：{}".format(e), 500)

    async def api_delete_summary(self, filename):
        filepath = self._summary_path(filename)
        if filepath is None or not filepath.exists() or not filepath.is_file():
            return _json_err("未找到总结文件", 404)
        try:
            doc_id = self._find_summary_doc_id(filename)
            filepath.unlink()
            removed_doc = False
            if doc_id and doc_id in self.memory_store.documents:
                removed_doc = self.memory_store.remove_document(doc_id)
                self._remove_deleted_doc_assignments(doc_id)
            self._remove_summary_record_by_filename(filename)
            return _json_ok({"deleted": filename, "memory_doc_id": doc_id, "memory_deleted": removed_doc})
        except Exception as e:
            return _json_err("删除总结失败：{}".format(e), 500)

    def _summary_path(self, filename):
        filename = Path(str(filename)).name
        if not SUMMARY_NAME_RE.match(filename):
            return None
        filepath = (self.summaries_dir / filename).resolve()
        try:
            filepath.relative_to(self.summaries_dir.resolve())
        except ValueError:
            return None
        return filepath

    async def terminate(self):
        logger.info("[{}] 插件已停止".format(PLUGIN_DISPLAY_NAME))

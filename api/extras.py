"""
api/extras.py
=============
The places added after Odysseus' review (2026-10-21): Compare (blind model arena), Find models (Hugging Face,
sized for this PC), Notes & tasks, Docs and Mail. Kept out of server.py, which includes this router.

Model calls here (compare answers, document and mail help) take the app's turn lock without waiting: one local
model runs at a time, so a request while a chat reply is being written answers "busy" instead of swapping the
model out from under it.
"""
from __future__ import annotations

import asyncio
import threading
import time
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

import agent.models as llm
import agent_skills
import compare_arena
import connectors
import knowledge
import profile_store
import core.config as config
import core.state as shared
import docs_store
import events as event_log
import llama_local
import mail
import model_finder
import notes

router = APIRouter()
BUSY = "Busy - wait for the current reply (or model load) to finish, then try again."


def _err(e: Exception) -> dict:
    msg = e.args[0] if e.args else str(e)
    return {"ok": False, "message": str(msg)[:400]}


def _with_model(fn):
    """Run fn() holding the turn lock (not waiting for it)."""
    if not shared.state_lock.acquire(blocking=False):
        return {"ok": False, "busy": True, "message": BUSY}
    try:
        return fn()
    except Exception as e:
        return _err(e)
    finally:
        shared.state_lock.release()


def _text_model(system, user, max_tokens=1500, model=None, label="Model is writing"):
    # A named local model with llama-server down waited out the 300 s timeout - say so at once instead.
    if model and model.startswith("local:") and llama_local.server_models(force=True) is None:
        raise RuntimeError(llama_local.unreachable_message())
    return llm.model_text(system, user, max_tokens=max_tokens, model=model, label=label)


# ==== Compare ===================================================================================================
def model_choices() -> list[dict]:
    out = []
    try:
        names = [c["name"] for c in llama_local.list_model_configs()]
    except Exception:
        names = []
    for n in names:
        out.append({"id": f"local:{n}", "name": n, "kind": "local"})
    for m in config.PRIMARY_MODELS:
        try:
            if llm.get_keys_for_model(m):
                out.append({"id": m, "name": m.split("/")[-1], "kind": "cloud"})
        except Exception:
            pass
    return out


class CompareStart(BaseModel):
    prompt: str
    a: str
    b: str
    blind: bool = True
    system: str = ""


class Vote(BaseModel):
    winner: str


class Combine(BaseModel):
    model: str | None = None


@router.get("/api/compare")
def compare_list():
    return {"rounds": compare_arena.rounds(), "leaderboard": compare_arena.leaderboard(), "models": model_choices()}


@router.post("/api/compare")
def compare_start(body: CompareStart):
    if not shared.state_lock.acquire(blocking=False):
        return {"ok": False, "busy": True, "message": BUSY}
    try:
        r = compare_arena.start(body.prompt, body.a, body.b, body.blind, body.system,
                                answer=lambda s, u, m: _text_model(s, u, 2048, m, "Compare: answering"),
                                lock=shared.state_lock)
    except Exception as e:
        shared.state_lock.release()
        return _err(e)
    event_log.log_action("Compare: a new round started.")
    return {"ok": True, "round": r}


@router.get("/api/compare/{rid}")
def compare_get(rid: str):
    r = compare_arena.get(rid)
    return {"ok": True, "round": r} if r else {"ok": False, "message": "No such round."}


@router.post("/api/compare/{rid}/vote")
def compare_vote(rid: str, body: Vote):
    try:
        return {"ok": True, "round": compare_arena.vote(rid, body.winner), "leaderboard": compare_arena.leaderboard()}
    except (KeyError, ValueError) as e:
        return _err(e)


@router.post("/api/compare/{rid}/combine")
def compare_combine(rid: str, body: Combine):
    return _with_model(lambda: {"ok": True, "round": compare_arena.combine(
        rid, lambda s, u, m: _text_model(s, u, 2048, m, "Compare: combining"), body.model)})


@router.delete("/api/compare/{rid}")
def compare_delete(rid: str):
    return {"ok": compare_arena.delete(rid), "leaderboard": compare_arena.leaderboard()}


# ==== Find models ===============================================================================================
class FinderDownload(BaseModel):
    repo: str
    path: str
    mmproj: bool = True


@router.get("/api/model-finder/search")
async def finder_search(q: str = ""):
    try:
        rows, hw = await asyncio.gather(asyncio.to_thread(model_finder.search, q), asyncio.to_thread(model_finder.hardware))
        return {"ok": True, "results": rows, "hardware": hw}
    except Exception as e:
        return _err(e)


@router.get("/api/model-finder/files")
async def finder_files(repo: str):
    try:
        return {"ok": True, **await asyncio.to_thread(model_finder.files, repo)}
    except Exception as e:
        return _err(e)


@router.post("/api/model-finder/download")
async def finder_download(body: FinderDownload):
    try:
        return await asyncio.to_thread(model_finder.download, body.repo, body.path, body.mmproj)
    except Exception as e:
        return _err(e)


@router.get("/api/model-finder/starter")
async def finder_starter():
    """The first-run welcome: the starter model that suits this PC (no network)."""
    try:
        return {"ok": True, **await asyncio.to_thread(model_finder.starter)}
    except Exception as e:
        return _err(e)


@router.get("/api/model-finder/status")
def finder_status():
    return model_finder.status()


@router.post("/api/model-finder/cancel")
def finder_cancel():
    return model_finder.cancel()


@router.post("/api/model-finder/pause")
def finder_pause():
    return model_finder.pause()


# ==== Notes & tasks =============================================================================================
class NoteBody(BaseModel):
    body: str = ""
    title: str | None = None
    pinned: bool | None = None


class TaskBody(BaseModel):
    text: str | None = None
    due: str | None = None
    done: bool | None = None
    note_id: str | None = None


@router.get("/api/notes")
def notes_all():
    return notes.all_data()


@router.post("/api/notes")
def note_create(body: NoteBody):
    try:
        return {"ok": True, "note": notes.save_note(None, body.body, body.title, body.pinned)}
    except (KeyError, ValueError) as e:
        return _err(e)


@router.post("/api/notes/{nid}")
def note_update(nid: str, body: NoteBody):
    try:
        return {"ok": True, "note": notes.save_note(nid, body.body, body.title, body.pinned)}
    except (KeyError, ValueError) as e:
        return _err(e)


@router.delete("/api/notes/{nid}")
def note_delete(nid: str):
    return {"ok": notes.delete_note(nid)}


@router.post("/api/tasks")
def task_create(body: TaskBody):
    try:
        return {"ok": True, "task": notes.add_task(body.text or "", body.due or "", body.note_id)}
    except (KeyError, ValueError) as e:
        return _err(e)


@router.post("/api/tasks/clear-done")
def task_clear_done():
    return {"ok": True, "removed": notes.clear_done()}


@router.post("/api/tasks/{tid}")
def task_update(tid: str, body: TaskBody):
    try:
        return {"ok": True, "task": notes.update_task(tid, text=body.text, due=body.due, done=body.done)}
    except (KeyError, ValueError) as e:
        return _err(e)


@router.delete("/api/tasks/{tid}")
def task_delete(tid: str):
    return {"ok": notes.delete_task(tid)}


# ==== Docs ======================================================================================================
class DocBody(BaseModel):
    content: str | None = None
    title: str | None = None
    summary: str = "Edit"


class DocAi(BaseModel):
    selection: str = ""
    instruction: str = ""
    before: str = ""
    after: str = ""


@router.get("/api/docs")
def docs_list():
    return {"docs": docs_store.list_docs()}


@router.post("/api/docs")
def docs_create(body: DocBody):
    try:
        return {"ok": True, "doc": docs_store.create(body.content or "", body.title or "")}
    except ValueError as e:
        return _err(e)


@router.get("/api/docs/{did}")
def docs_get(did: str):
    try:
        return {"ok": True, "doc": docs_store.get(did)}
    except KeyError as e:
        return _err(e)


@router.post("/api/docs/{did}")
def docs_save(did: str, body: DocBody):
    try:
        d = docs_store.save(did, body.content, body.title, body.summary or "Edit")
        return {"ok": True, "doc": {k: v for k, v in d.items() if k != "content"}}
    except (KeyError, ValueError) as e:
        return _err(e)


@router.delete("/api/docs/{did}")
def docs_delete(did: str):
    try:
        return {"ok": docs_store.delete(did)}
    except KeyError as e:
        return _err(e)


@router.get("/api/docs/{did}/versions")
def docs_versions(did: str):
    try:
        return {"ok": True, "versions": docs_store.versions(did)}
    except KeyError as e:
        return _err(e)


@router.post("/api/docs/{did}/restore/{n}")
def docs_restore(did: str, n: int):
    try:
        return {"ok": True, "doc": docs_store.restore(did, n)}
    except KeyError as e:
        return _err(e)


@router.post("/api/docs/ai/edit")
def docs_ai_edit(body: DocAi):
    return _with_model(lambda: {"ok": True, **docs_store.ai_edit(_text_model, body.selection, body.instruction, body.before, body.after)})


@router.post("/api/docs/ai/proofread")
def docs_ai_proofread(body: DocAi):
    return _with_model(lambda: {"ok": True, "suggestions": docs_store.proofread(llm.model_json, body.selection)})


# ==== Mail ======================================================================================================
class MailAccount(BaseModel):
    email: str
    password: str = ""
    name: str = ""
    username: str = ""
    imap_host: str = ""
    imap_port: int = 0
    smtp_host: str = ""
    smtp_port: int = 0
    smtp_security: str = ""
    signature: str = ""


class MailSend(BaseModel):
    account: str | None = None
    to: str = ""
    cc: str = ""
    bcc: str = ""
    subject: str = ""
    body: str = ""
    in_reply_to: str = ""
    references: str = ""
    draft_id: str | None = None


class MailAct(BaseModel):
    account: str | None = None
    folder: str = "INBOX"
    uids: list[int] = []
    flag: str = ""
    on: bool = True
    to_role: str = ""
    to_folder: str = ""


class MailAi(BaseModel):
    account: str | None = None
    folder: str = "INBOX"
    uid: int = 0
    instruction: str = ""


class MailDraft(BaseModel):
    account: str | None = None
    to: str = ""
    cc: str = ""
    subject: str = ""
    body: str = ""
    in_reply_to: str = ""
    references: str = ""


def sync_provider() -> bool:
    return mail.register(shared.messaging_hub)


@router.get("/api/mail/accounts")
def mail_accounts():
    return {"accounts": mail.accounts(), "drafts": mail.drafts()}


@router.get("/api/mail/preset")
def mail_preset(email: str = ""):
    return mail.preset(email)


@router.post("/api/mail/accounts")
async def mail_add(body: MailAccount):
    try:
        a = await asyncio.to_thread(mail.save_account, body.model_dump(), None, True)
    except Exception as e:
        return _err(e)
    sync_provider()
    event_log.log_action(f"Mail: added {a['email']}.")
    return {"ok": True, "account": a}


@router.post("/api/mail/accounts/{aid}")
async def mail_edit(aid: str, body: MailAccount):
    try:
        a = await asyncio.to_thread(mail.save_account, body.model_dump(), aid, True)
    except Exception as e:
        return _err(e)
    return {"ok": True, "account": a}


@router.delete("/api/mail/accounts/{aid}")
def mail_remove(aid: str):
    ok = mail.remove_account(aid)
    sync_provider()
    return {"ok": ok}


@router.get("/api/mail/folders")
async def mail_folders(account: str = ""):
    try:
        return {"ok": True, "folders": await asyncio.to_thread(mail.folders, account or None)}
    except Exception as e:
        return _err(e)


@router.get("/api/mail/list")
async def mail_list(account: str = "", folder: str = "INBOX", before: int = 0, q: str = "", unread: bool = False):
    try:
        return {"ok": True, **await asyncio.to_thread(mail.list_messages, account or None, folder, 50, before or None, q, unread)}
    except Exception as e:
        return _err(e)


@router.get("/api/mail/message")
async def mail_message(uid: int, account: str = "", folder: str = "INBOX"):
    try:
        return {"ok": True, "message": await asyncio.to_thread(mail.read, account or None, folder, uid, True)}
    except Exception as e:
        return _err(e)


@router.get("/api/mail/attachment")
def mail_attachment(uid: int, index: int, account: str = "", folder: str = "INBOX"):
    try:
        name, ctype, data = mail.attachment(account or None, folder, uid, index)
    except Exception as e:
        return Response(str(e)[:200], status_code=404)
    # Always a download, never shown inline: an attachment's HTML or SVG must not run in the app's origin.
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}",
                             "X-Content-Type-Options": "nosniff"})


@router.post("/api/mail/act")
async def mail_act(body: MailAct):
    try:
        if body.flag:
            return await asyncio.to_thread(mail.set_flag, body.account, body.folder, body.uids, body.flag, body.on)
        return await asyncio.to_thread(mail.move, body.account, body.folder, body.uids, body.to_role, body.to_folder)
    except Exception as e:
        return _err(e)


@router.post("/api/mail/send")
async def mail_send(body: MailSend):
    """Pressed Send in the Mail place: the user's own explicit action."""
    try:
        out = await asyncio.to_thread(mail.send, body.account, body.to, body.subject, body.body, body.cc, body.bcc,
                                      body.in_reply_to, body.references)
    except Exception as e:
        return _err(e)
    if body.draft_id:
        mail.delete_draft(body.draft_id)
    event_log.log_action(f"Mail: sent \"{body.subject[:80]}\" to {body.to[:120]}.")
    return out


@router.post("/api/mail/drafts")
def mail_draft_save(body: MailDraft, id: str = ""):
    d = body.model_dump()
    if id:
        d["id"] = id
    return {"ok": True, "draft": mail.save_draft(d)}


@router.delete("/api/mail/drafts/{did}")
def mail_draft_delete(did: str):
    return {"ok": mail.delete_draft(did)}


def _msg(body: MailAi) -> dict:
    return mail.read(body.account, body.folder, body.uid, mark_seen=False)


@router.post("/api/mail/ai/summary")
def mail_ai_summary(body: MailAi):
    return _with_model(lambda: {"ok": True, "summary": mail.summarize(_text_model, _msg(body))})


@router.post("/api/mail/ai/reply")
def mail_ai_reply(body: MailAi):
    def run():
        m = _msg(body)
        a = mail.account(m["account"])
        text = mail.draft_reply(_text_model, m, body.instruction, a.get("name") or shared.app_state["settings"].get("user_name") or "")
        return {"ok": True, "reply": {**mail.reply_fields(m), "body": text, "account": m["account"]}}
    return _with_model(run)


@router.post("/api/mail/ai/triage")
def mail_ai_triage(body: MailAi):
    def run():
        lst = mail.list_messages(body.account, body.folder, 25)
        return {"ok": True, "tags": mail.triage(llm.model_json, lst["messages"])}
    return _with_model(run)


# ==== Memory: import / export (memory.py) =========================================================================
class MemoryImport(BaseModel):
    data: dict | list | str


@router.get("/api/memory-export")
def memory_export():
    import json
    return Response(json.dumps(shared.memory_store.export(), ensure_ascii=False, indent=1), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=fx-memory.json"})


@router.post("/api/memory-import")
def memory_import(body: MemoryImport):
    try:
        r = shared.memory_store.import_entries(body.data)
    except Exception as e:
        return _err(e)
    event_log.log_action(f"Memory: imported {r['added']} entries ({r['skipped']} skipped).")
    return {"ok": True, **r}


# ==== About you (profile_store.py) =================================================================================
class ProfileBody(BaseModel):
    text: str | None = None


@router.get("/api/profile")
def profile_get():
    return {**profile_store.get(), "auto": shared.app_state["settings"].get("profile_auto", True)}


@router.post("/api/profile")
def profile_set(body: ProfileBody):
    try:
        return {"ok": True, **profile_store.set_text(body.text or "")}
    except Exception as e:
        return _err(e)


@router.post("/api/profile/propose")
def profile_propose():
    return _with_model(lambda: {"ok": True, **profile_store.propose(
        _text_model, shared.chat_store, bool(shared.app_state["settings"].get("memory_sensitive")))})


@router.post("/api/profile/accept")
def profile_accept(body: ProfileBody):
    try:
        return {"ok": True, **profile_store.accept(body.text)}
    except ValueError as e:
        return _err(e)


@router.post("/api/profile/discard")
def profile_discard():
    return {"ok": True, **profile_store.discard()}


# ==== Skills (agent_skills.py) =====================================================================================
class SkillBody(BaseModel):
    name: str = ""
    description: str = ""
    body: str = ""
    enabled: bool | None = None


class SkillDraft(BaseModel):
    task_id: str = ""
    request: str = ""
    notes: str = ""


class SkillImport(BaseModel):
    filename: str = "SKILL.md"
    data_b64: str = ""
    text: str = ""


@router.get("/api/skills")
def skills_list():
    return {"skills": agent_skills.list_skills(), "folder": agent_skills.ROOT,
            "offer": shared.app_state["settings"].get("skills_offer", True)}


@router.get("/api/skills/{slug}")
def skills_get(slug: str):
    try:
        s = agent_skills.get(slug)
        return {"ok": True, "skill": s, "review": agent_skills.review(agent_skills.render(
            {"name": s["name"], "description": s["description"]}, s["body"]), s["files"])}
    except KeyError as e:
        return _err(e)


@router.post("/api/skills")
def skills_create(body: SkillBody):
    try:
        s = agent_skills.save(body.name, body.description, body.body, source="user", enabled=body.enabled)
    except (KeyError, ValueError) as e:
        return _err(e)
    event_log.log_action(f"Skill saved: {s['name']}.")
    return {"ok": True, "skill": s}


@router.post("/api/skills/{slug}")
def skills_update(slug: str, body: SkillBody):
    try:
        if body.name or body.body:
            s = agent_skills.save(body.name, body.description, body.body, slug=slug, enabled=body.enabled)
        else:
            s = agent_skills.set_enabled(slug, bool(body.enabled))
    except (KeyError, ValueError) as e:
        return _err(e)
    return {"ok": True, "skill": s}


@router.delete("/api/skills/{slug}")
def skills_delete(slug: str):
    try:
        return {"ok": agent_skills.delete(slug)}
    except KeyError as e:
        return _err(e)


@router.get("/api/skills/{slug}/export")
def skills_export(slug: str):
    try:
        data = agent_skills.export_zip(slug)
    except KeyError as e:
        return Response(str(e), status_code=404)
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={slug}.zip"})


@router.post("/api/skills-import")
def skills_import(body: SkillImport):
    import base64
    try:
        raw = base64.b64decode(body.data_b64) if body.data_b64 else body.text.encode("utf-8")
        if len(raw) > 25 * 2 ** 20:
            raise ValueError("That file is over 25 MB.")
        s = agent_skills.import_skill(raw, body.filename or "SKILL.md")
    except Exception as e:
        return _err(e)
    event_log.log_action(f"Skill imported (off until you turn it on): {s['name']}.")
    return {"ok": True, "skill": s}


@router.post("/api/skills-draft")
def skills_draft(body: SkillDraft):
    """A draft (not saved) from a run that worked (task_id - the run's replayable steps) or from what the user wants
    the skill to cover. The page shows it to edit, then POST /api/skills keeps it."""
    def run():
        import agent.routine_runner as routine_runner
        cand = routine_runner._candidates.get(body.task_id) if body.task_id else None
        if body.task_id and not cand:
            raise ValueError("That run is too old to learn from - ask for the skill in a chat instead.")
        if cand and cand.get("tainted"):
            raise ValueError("That run read content that tried to instruct the assistant, so it isn't made into a skill.")
        request = (cand or {}).get("request") or body.request
        if not request and not body.notes:
            raise ValueError("Say what the skill should cover.")
        return {"ok": True, "draft": agent_skills.draft(llm.model_json, request or body.notes, (cand or {}).get("steps"), body.notes)}
    return _with_model(run)


# ==== Knowledge base (knowledge.py) ================================================================================
class KnowledgeBody(BaseModel):
    folder: str = ""


def _kb_folders() -> list:
    return list(shared.app_state["settings"].get("knowledge_folders") or [])


def _kb_save(folders: list) -> None:
    current = shared.app_state["settings"]
    current["knowledge_folders"] = folders
    current.update(config.clean_agent_settings(current))
    config.save_settings(current)


@router.get("/api/knowledge")
def knowledge_status():
    st = knowledge.status()
    return {"folders": [{"path": f, **st["folders"].get(__import__("os").path.abspath(f), {"files": 0, "passages": 0, "skipped": 0}),
                         "exists": __import__("os").path.isdir(f)} for f in _kb_folders()],
            **{k: v for k, v in st.items() if k != "folders"}}


@router.post("/api/knowledge/folders")
def knowledge_add(body: KnowledgeBody):
    import os
    f = os.path.abspath(os.path.expandvars(body.folder.strip().strip('"')))
    if not body.folder.strip() or not os.path.isdir(f):
        return {"ok": False, "message": "There's no folder at that path."}
    if len(os.path.splitdrive(f)[1].strip("\\/")) == 0:
        return {"ok": False, "message": "Pick a folder, not a whole drive - it would index everything on it."}
    folders = _kb_folders()
    if f not in folders:
        folders.append(f)
        _kb_save(folders)
    knowledge.start(folders)
    return {"ok": True}


@router.post("/api/knowledge/remove")
def knowledge_remove(body: KnowledgeBody):
    folders = [x for x in _kb_folders() if x != body.folder]
    _kb_save(folders)
    knowledge.forget(body.folder)
    return {"ok": True}


@router.post("/api/knowledge/reindex")
def knowledge_reindex():
    return knowledge.start(_kb_folders())


@router.get("/api/knowledge/search")
def knowledge_search(q: str = ""):
    return {"ok": True, "results": [{k: v for k, v in h.items() if k != "text"} for h in knowledge.search(q, 10)]}


# ==== One-click connectors (connectors.py) =========================================================================
class ConnectorBody(BaseModel):
    values: dict = {}


@router.get("/api/connectors")
def connectors_list():
    return {"connectors": connectors.catalog(shared.mcp_hub), "node": connectors.node_ok()}


@router.post("/api/connectors/{cid}")
def connectors_install(cid: str, body: ConnectorBody):
    try:
        return connectors.install(cid, body.values or {}, shared.mcp_hub, event_log.log_action)
    except (KeyError, ValueError) as e:
        return _err(e)


@router.delete("/api/connectors/{cid}")
def connectors_remove(cid: str):
    return {"ok": connectors.remove(cid, shared.mcp_hub)}


# ==== background: due tasks ======================================================================================
_started = False


def _auto_profile() -> None:
    """After enough new chats, propose an updated profile - only while nothing else uses the model and a local
    model is already loaded (it never loads one just for this). The proposal waits for the user."""
    s = shared.app_state["settings"]
    if not s.get("profile_auto", True) or not s.get("memory_enabled", True):
        return
    try:
        if not profile_store.due_for_auto(shared.chat_store):
            return
        if s.get("mode") == config.MODE_LOCAL or shared.app_state.get("mode") == config.MODE_LOCAL:
            if not shared.app_state.get("local_engaged"):
                return
        if not shared.state_lock.acquire(blocking=False):
            return
        try:
            profile_store.propose(_text_model, shared.chat_store, bool(s.get("memory_sensitive")), only_new=False)
            event_log.log_action("Your profile: a new version is waiting in Settings > Memory.")
        finally:
            shared.state_lock.release()
    except Exception as e:
        event_log.log_action(f"Couldn't propose a profile update: {str(e)[:120]}")


def start_background():
    global _started
    if _started:
        return
    _started = True
    try:
        compare_arena.recover()
    except Exception:
        pass
    try:
        sync_provider()
    except Exception as e:
        event_log.log_action(f"Mail: couldn't set up the assistant's email tools: {e}")

    # The knowledge base catches up with files changed while the app was closed.
    if _kb_folders():
        knowledge.start(_kb_folders())

    def loop():
        import productivity_tools
        n = 0
        while True:
            try:
                for t in notes.due_check(productivity_tools.send_notification):
                    event_log.log_action(f"Task due: {t['text'][:120]}")
            except Exception:
                pass
            n += 1
            if n % 30 == 0:                  # every half hour: re-read changed files, maybe propose a new profile
                try:
                    if _kb_folders():
                        knowledge.start(_kb_folders())
                except Exception:
                    pass
                _auto_profile()
            time.sleep(60)
    threading.Thread(target=loop, name="tasks-due", daemon=True).start()

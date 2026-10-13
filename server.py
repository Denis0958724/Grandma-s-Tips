#!/usr/bin/env python3
"""
Поради бабусі — локальний сервер.

Відкриває сайт у браузері й передає запити ШІ до Claude API, до Ollama на цьому
комп'ютері (знаходить її сам) або до іншої OpenAI-сумісної моделі.
Ключ ШІ залишається тут, на сервері, і ніколи не потрапляє в браузер.

Запуск:  python server.py      (або двічі клацнути «Запустити з ШІ.bat»)
Налаштування: файл ai-settings.txt поруч із цим файлом.
Потрібен лише Python 3.9+, сторонні бібліотеки не потрібні.
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
SITE_FILE = 'Поради бабусі.html'
SETTINGS_FILE = 'ai-settings.txt'
EXAMPLE_FILE = 'ai-settings.example.txt'
MAX_BODY = 300_000          # байтів у запиті від сайту
MAX_TOKENS = 2000           # довжина відповіді моделі
TIMEOUT = 120               # секунд на відповідь моделі

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
ANTHROPIC_URL = os.environ.get('ANTHROPIC_URL', 'https://api.anthropic.com').rstrip('/')
OLLAMA_SEARCH_URL = os.environ.get('OLLAMA_SEARCH_URL', 'https://ollama.com/api/web_search')
# моделі, які краще пишуть українською й підходять для розмови (у порядку переваги)
PREFERRED = ('gemma3', 'qwen3', 'qwen2.5', 'llama3', 'mistral', 'gpt-oss', 'aya')
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def load_settings():
    """Читає ai-settings.txt (рядки КЛЮЧ=значення). Змінні середовища мають пріоритет."""
    path = os.path.join(HERE, SETTINGS_FILE)
    example = os.path.join(HERE, EXAMPLE_FILE)
    if not os.path.exists(path) and os.path.exists(example):
        # перший запуск після завантаження з GitHub: створюємо налаштування з шаблону
        with open(example, encoding='utf-8-sig') as f:
            lines = f.read().splitlines(True)
        if lines and 'ШАБЛОН' in lines[0]:
            lines = ['# Налаштування ШІ для «Поради бабусі»\n'] + lines[3:]
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(''.join(lines))
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                value = value.strip().strip('"').strip("'")
                if value and not value.startswith('встав'):
                    os.environ.setdefault(key.strip(), value)

    anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
    want = os.environ.get('AI_PROVIDER', '').lower()
    cfg = {'port': int(os.environ.get('PORT', '8000'))}
    if want == 'openai' and os.environ.get('OPENAI_BASE_URL'):
        model = os.environ.get('OPENAI_MODEL', '')
        cfg.update(provider='openai', key=os.environ.get('OPENAI_API_KEY', ''),
                   base=os.environ['OPENAI_BASE_URL'].rstrip('/'), model=model,
                   model_quick=os.environ.get('OPENAI_MODEL_QUICK', model), configured=bool(model))
        cfg['hint'] = '' if model else 'Вкажи OPENAI_MODEL у ' + SETTINGS_FILE + '.'
        return cfg
    if anthropic_key and want in ('', 'anthropic'):
        cfg.update(provider='anthropic', key=anthropic_key, configured=True, hint='',
                   model=os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-5-5'),
                   model_quick=os.environ.get('ANTHROPIC_MODEL_QUICK', 'claude-haiku-4-5-20251001'))
        return cfg
    cfg.update(detect_ollama())
    return cfg


def ollama_models():
    # Список завантажених моделей Ollama або None, якщо Ollama не запущено
    for path, key, field in (('/api/tags', 'models', 'name'), ('/v1/models', 'data', 'id')):
        try:
            with _NO_PROXY.open(OLLAMA_URL + path, timeout=2) as r:
                data = json.loads(r.read().decode('utf-8'))
            names = [m.get(field) or m.get('model') for m in data.get(key, [])]
            return [n for n in names if n and 'embed' not in n.lower()]
        except Exception:
            continue
    return None


def detect_ollama():
    names = ollama_models()
    if names is None:
        return {'provider': 'none', 'configured': False,
                'hint': 'ШІ не знайдено. Запусти Ollama й завантаж модель (ollama pull gemma3:4b) або впиши ключ Claude API у ' + SETTINGS_FILE + '.'}
    if not names:
        return {'provider': 'ollama', 'configured': False,
                'hint': 'Ollama працює, але моделей ще немає. Виконай у терміналі: ollama pull gemma3:4b'}
    wanted = os.environ.get('OLLAMA_MODEL', '') or (os.environ.get('OPENAI_MODEL', '') if os.environ.get('AI_PROVIDER', '').lower() == 'openai' else '')
    pick = None
    if wanted:
        pick = next((n for n in names if n == wanted or n == wanted + ':latest'), None)
    if not pick:
        talk = [n for n in names if 'coder' not in n.lower()] or names
        pick = next((n for p in PREFERRED for n in talk if n.lower().startswith(p)), talk[0])
    return {'provider': 'ollama', 'configured': True, 'hint': '', 'key': '', 'base': OLLAMA_URL + '/v1',
            'model': pick, 'model_quick': pick, 'all_models': names}


CFG = load_settings()


class AIError(Exception):
    def __init__(self, code, message, status=502):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def open_upstream(messages, system, tier):
    """Відкриває потокове з'єднання з моделлю й повертає генератор шматочків тексту."""
    quick = tier == 'quick'
    if CFG['provider'] == 'anthropic':
        body = {'model': CFG['model_quick'] if quick else CFG['model'], 'max_tokens': MAX_TOKENS,
                'messages': messages, 'stream': True}
        if system:
            body['system'] = system
        req = urllib.request.Request(ANTHROPIC_URL + '/v1/messages', data=json.dumps(body).encode('utf-8'),
                                     headers={'x-api-key': CFG['key'], 'anthropic-version': '2023-06-01',
                                              'content-type': 'application/json'})
        resp = _urlopen(req)

        def gen():
            for raw in resp:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                if ev.get('type') == 'content_block_delta' and ev.get('delta', {}).get('type') == 'text_delta':
                    yield ev['delta'].get('text', '')
                elif ev.get('type') == 'error':
                    raise AIError('upstream_error', ev.get('error', {}).get('message', 'помилка моделі'))
        return gen()

    # OpenAI-сумісний API: Ollama, LM Studio, OpenRouter тощо (міркування моделей у <think> відкидаємо)
    msgs = ([{'role': 'system', 'content': system}] if system else []) + messages
    body = {'model': CFG['model_quick'] if quick else CFG['model'], 'messages': msgs, 'stream': True, 'max_tokens': MAX_TOKENS}
    headers = {'content-type': 'application/json'}
    if CFG.get('key'):
        headers['authorization'] = 'Bearer ' + CFG['key']
    req = urllib.request.Request(CFG['base'] + '/chat/completions', data=json.dumps(body).encode('utf-8'), headers=headers)
    resp = _urlopen(req)

    def gen():
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                break
            try:
                ev = json.loads(payload)
            except ValueError:
                continue
            choices = ev.get('choices') or [{}]
            piece = (choices[0].get('delta') or {}).get('content')
            if piece:
                yield piece
    return strip_think(gen())


def strip_think(chunks):
    # Прибирає з потоку блоки <think>…</think>, які пишуть деякі моделі (qwen3, deepseek-r1)
    buf, inside = '', False
    for piece in chunks:
        buf += piece
        out = ''
        while True:
            if inside:
                i = buf.find('</think>')
                if i < 0:
                    buf = buf[-8:]
                    break
                buf, inside = buf[i + 8:].lstrip(), False
            else:
                i = buf.find('<think>')
                if i < 0:
                    keep = next((k for k in range(min(6, len(buf)), 0, -1) if '<think>'.startswith(buf[-k:])), 0)
                    out += buf[:len(buf) - keep]
                    buf = buf[len(buf) - keep:]
                    break
                out += buf[:i]
                buf, inside = buf[i + 7:], True
        if out:
            yield out
    if buf and not inside:
        yield buf


def _urlopen(req, who='ШІ'):
    try:
        # локальні моделі (Ollama, LM Studio) — напряму, без системного проксі
        if urlsplit(req.full_url).hostname in ('127.0.0.1', 'localhost'):
            return _NO_PROXY.open(req, timeout=TIMEOUT)
        return urllib.request.urlopen(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            detail = json.loads(e.read().decode('utf-8', 'replace')).get('error', {})
            detail = detail.get('message', '') if isinstance(detail, dict) else str(detail)
        except Exception:
            pass
        if e.code in (401, 403):
            raise AIError('bad_key', 'Ключ ' + who + ' не підходить. Перевір ' + SETTINGS_FILE + '.', 401)
        if e.code == 429:
            raise AIError('rate_limited', 'Забагато запитів або скінчився баланс. Спробуй пізніше.', 429)
        if e.code == 404:
            raise AIError('upstream_error', 'Модель не знайдено: ' + (detail or 'перевір назву моделі в ' + SETTINGS_FILE), 502)
        raise AIError('upstream_error', 'Модель повернула помилку %s. %s' % (e.code, detail), 502)
    except urllib.error.URLError as e:
        if who != 'ШІ':
            raise AIError('upstream_error', 'Немає зв’язку з пошуком. Перевір інтернет.', 502)
        hint = ' Чи запущено Ollama?' if CFG['provider'] in ('ollama', 'openai') else ' Перевір інтернет.'
        raise AIError('upstream_error', 'Немає зв’язку з моделлю.' + hint, 502)


# ---------- Розумний пошук в інтернеті ----------
# Два шляхи: Claude API з інструментом web_search (Claude сам шукає й відбирає)
# або вебпошук Ollama (безкоштовний акаунт) + будь-яка модель для відбору.
# Посилання в результатах беруться ЛИШЕ з того, що реально повернув пошук.
KIND_WORDS = {'gift': 'подарунки', 'service': 'послуги (майстер-класи, масаж, фотосесії, доставка квітів)', 'experience': 'враження (концерти, театр, екскурсії, ресторани, поїздки)'}


def search_mode():
    if CFG.get('provider') == 'anthropic' and CFG.get('key'):
        return 'claude'
    if os.environ.get('OLLAMA_API_KEY'):
        return 'ollama'
    return 'none'


def _norm_url(u):
    try:
        p = urlsplit(u.strip())
        if p.scheme not in ('http', 'https') or not p.netloc:
            return ''
        return (p.netloc.lower().removeprefix('www.') + p.path.rstrip('/') + ('?' + p.query if p.query else '')).lower()
    except Exception:
        return ''


def _store(u):
    return urlsplit(u).netloc.lower().removeprefix('www.')


def _loose_json(text):
    text = (text or '').strip()
    for cand in (text, text[text.find('{'):text.rfind('}') + 1] if '{' in text else ''):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _clip(v, n):
    return str(v or '').strip()[:n]


def search_task_text(task):
    lines = ['Сьогодні: ' + task.get('today', '')]
    if task.get('context'):
        lines.append('Що шукає користувач: ' + task['context'])
    kinds = [KIND_WORDS[k] for k in task.get('kinds', []) if k in KIND_WORDS] or list(KIND_WORDS.values())
    lines.append('Типи пропозицій: ' + '; '.join(kinds))
    if task.get('budget'):
        lines.append('Бюджет: ' + task['budget'])
    if task.get('city'):
        lines.append('Місто для послуг і вражень: ' + task['city'])
    if task.get('profile'):
        lines.append('Профіль коханої людини (JSON): ' + json.dumps(task['profile'], ensure_ascii=False)[:4000])
    if task.get('exclude'):
        lines.append('Не повторюй уже показане: ' + '; '.join(task['exclude'][:20]))
    return '\n'.join(lines)


def complete(prompt, as_json=False):
    # Повна (не потокова) відповідь налаштованої моделі
    system = 'Відповідай лише одним валідним JSON-значенням, без жодного тексту до чи після.' if as_json else ''
    return ''.join(open_upstream([{'role': 'user', 'content': prompt}], system, 'quick'))


def claude_search(task):
    loc = {'type': 'approximate', 'country': 'UA', 'timezone': 'Europe/Kyiv'}
    if task.get('city'):
        loc['city'] = task['city']
    tools = [{'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': 5, 'user_location': loc}]
    prompt = ("""Ти — помічниця застосунку «Поради бабусі». Знайди в інтернеті конкретні пропозиції, які можна купити або замовити в Україні (з доставкою в Україну), щоб потішити кохану людину користувача.
Шукай справжні сторінки товарів і послуг в українських магазинах і сервісах. Враховуй інтереси, улюблене, мову любові, алергії й те, що людина не любить. Дотримуйся бюджету.

""" + search_task_text(task) + """

Після пошуку поверни лише JSON без жодного іншого тексту:
{"items":[{"title":"коротка назва","url":"точне посилання з результатів пошуку","store":"назва магазину чи сервісу","price":"ціна, якщо вона є на сторінці, інакше порожньо","why":"чому підходить саме цій людині, до 20 слів","kind":"gift|service|experience"}]}
6–8 пропозицій. Посилання бери лише з результатів пошуку, нічого не вигадуй.""")
    messages = [{'role': 'user', 'content': prompt}]
    found, texts = {}, []
    for _ in range(3):
        body = {'model': CFG['model'], 'max_tokens': 4000, 'messages': messages, 'tools': tools}
        req = urllib.request.Request(ANTHROPIC_URL + '/v1/messages', data=json.dumps(body).encode('utf-8'),
                                     headers={'x-api-key': CFG['key'], 'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
        with _urlopen(req) as r:
            resp = json.loads(r.read().decode('utf-8'))
        for block in resp.get('content', []):
            if block.get('type') == 'web_search_tool_result' and isinstance(block.get('content'), list):
                for res in block['content']:
                    if res.get('type') == 'web_search_result' and _norm_url(res.get('url', '')):
                        found[_norm_url(res['url'])] = res['url']
            elif block.get('type') == 'text':
                texts.append(block.get('text', ''))
        if resp.get('stop_reason') == 'pause_turn':
            messages = [messages[0], {'role': 'assistant', 'content': resp['content']}]
            continue
        break
    data = None
    for t in [texts[-1] if texts else ''] + [''.join(texts)]:
        data = _loose_json(t)
        if data:
            break
    raw = data.get('items', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    items = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        key = _norm_url(it.get('url', ''))
        if key not in found:
            continue  # посилання не з результатів пошуку — відкидаємо
        url = found[key]
        items.append({'title': _clip(it.get('title'), 120), 'url': url, 'store': _clip(it.get('store'), 60) or _store(url),
                      'price': _clip(it.get('price'), 40), 'why': _clip(it.get('why'), 240), 'kind': it.get('kind') if it.get('kind') in KIND_WORDS else 'gift'})
    return {'source': 'claude', 'items': items[:10], 'checked': len(found)}


def ollama_web_search(query, n=6):
    req = urllib.request.Request(OLLAMA_SEARCH_URL, data=json.dumps({'query': query, 'max_results': n}).encode('utf-8'),
                                 headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', ''), 'Content-Type': 'application/json'})
    with _urlopen(req, who='пошуку Ollama') as r:
        return json.loads(r.read().decode('utf-8')).get('results', [])


def heuristic_queries(task):
    prof = task.get('profile') or {}
    base = task.get('context') or ''
    interests = prof.get('інтереси') or []
    kinds = task.get('kinds') or ['gift']
    qs = []
    if base:
        qs.append(base + ' купити')
    for i in interests[:2]:
        qs.append('подарунок для людини, яка любить ' + str(i).lower() + ' купити Україна')
    if 'experience' in kinds or 'service' in kinds:
        qs.append(('сертифікат враження ' + (task.get('city') or 'Україна')).strip())
    return (qs or ['оригінальний подарунок коханій людині купити Україна'])[:3]


def ollama_search(task):
    llm = CFG.get('configured')
    queries = []
    if llm:
        try:
            q = _loose_json(complete('Склади 3 різні короткі пошукові запити українською для пошуку в інтернет-магазинах і сервісах України за цим завданням. Кожен запит — до 8 слів, конкретний товар чи послуга.\n\n'
                                     + search_task_text(task) + '\n\nПоверни лише JSON: {"queries":["...","...","..."]}', as_json=True))
            queries = [str(x)[:120] for x in (q or {}).get('queries', []) if str(x).strip()][:3]
        except AIError:
            queries = []
    queries = queries or heuristic_queries(task)
    results, seen = [], set()
    for q in queries:
        for r in ollama_web_search(q):
            key = _norm_url(r.get('url', ''))
            if key and key not in seen:
                seen.add(key)
                results.append({'title': _clip(r.get('title'), 160), 'url': r['url'], 'snippet': _clip(r.get('content'), 400)})
    results = results[:18]
    items = []
    if llm and results:
        listing = '\n'.join('[%d] %s — %s — %s' % (i, r['title'], _store(r['url']), r['snippet'][:280]) for i, r in enumerate(results))
        try:
            pick = _loose_json(complete('Ось результати пошуку. Обери до 8, що найкраще підходять як подарунок, послуга чи враження для коханої людини, і поясни чому. '
                                        'Пропускай статті-огляди й сторінки без конкретної пропозиції, якщо є кращі. Ціну пиши лише якщо вона є в описі.\n\n'
                                        + search_task_text(task) + '\n\nРезультати:\n' + listing
                                        + '\n\nПоверни лише JSON: {"items":[{"i":0,"title":"коротка назва","why":"до 20 слів","price":"","kind":"gift|service|experience"}]}', as_json=True))
            for it in (pick or {}).get('items', []):
                try:
                    r = results[int(it.get('i'))]
                except (TypeError, ValueError, IndexError):
                    continue
                if any(x['url'] == r['url'] for x in items):
                    continue
                items.append({'title': _clip(it.get('title'), 120) or r['title'], 'url': r['url'], 'store': _store(r['url']),
                              'price': _clip(it.get('price'), 40), 'why': _clip(it.get('why'), 240), 'kind': it.get('kind') if it.get('kind') in KIND_WORDS else 'gift'})
        except AIError:
            items = []
    if not items:
        items = [{'title': r['title'], 'url': r['url'], 'store': _store(r['url']), 'price': '', 'why': r['snippet'][:200], 'kind': 'gift'} for r in results[:8]]
    return {'source': 'ollama', 'items': items[:10], 'queries': queries, 'checked': len(results)}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    # --- безпека: лише локальні запити й жодних службових файлів ---
    def _host_ok(self):
        host = (self.headers.get('Host') or '').split(':')[0]
        return host in ('127.0.0.1', 'localhost')

    def _blocked_path(self):
        path = urlsplit(self.path).path
        parts = [p for p in path.split('/') if p]
        return any(p.startswith('.') for p in parts) or any(SETTINGS_FILE in p or p.endswith('.py') or p.endswith('.bat') for p in parts)

    def _json(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        if urlsplit(self.path).path.endswith('.html'):
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def guess_type(self, path):
        t = super().guess_type(path)
        return t + '; charset=utf-8' if t in ('text/html', 'text/plain', 'text/markdown', 'application/json') else t

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {'code': 'forbidden'})
        path = urlsplit(self.path).path
        if path in ('/', '/index.html'):
            self.send_response(302)
            self.send_header('Location', '/' + quote(SITE_FILE))
            self.end_headers()
            return
        if path == '/api/health':
            if not CFG['configured']:
                CFG.update(load_settings())
                if CFG['configured']:
                    print('  ШІ знайдено: %s, модель %s' % (CFG['provider'], CFG.get('model', '')))
            return self._json(200, {'ok': True, 'provider': CFG['provider'], 'configured': CFG['configured'],
                                    'model': CFG.get('model', ''), 'message': CFG.get('hint', ''), 'search': search_mode()})
        if self._blocked_path():
            return self._json(404, {'code': 'not_found'})
        return super().do_GET()

    def do_POST(self):
        route = urlsplit(self.path).path
        if route not in ('/api/ai', '/api/search'):
            return self._json(404, {'code': 'not_found'})
        origin = self.headers.get('Origin')
        if not self._host_ok() or self.headers.get('X-Babusya') != '1' or (origin and urlsplit(origin).hostname not in ('127.0.0.1', 'localhost')):
            return self._json(403, {'code': 'forbidden', 'message': 'Запити приймаються лише від сайту на цьому комп’ютері.'})
        if route == '/api/search':
            return self._search()
        if not CFG['configured']:
            return self._json(503, {'code': 'not_configured', 'message': CFG.get('hint') or 'ШІ не налаштовано.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > MAX_BODY:
                raise ValueError('size')
            body = json.loads(self.rfile.read(length).decode('utf-8'))
            messages = [{'role': m['role'], 'content': str(m['content'])} for m in body['messages']
                        if m.get('role') in ('user', 'assistant') and str(m.get('content', '')).strip()][-40:]
            if not messages or messages[0]['role'] != 'user' or messages[-1]['role'] != 'user':
                raise ValueError('roles')
        except Exception:
            return self._json(400, {'code': 'invalid_request', 'message': 'Неправильний запит.'})
        system = 'Відповідай лише одним валідним JSON-значенням, без жодного тексту до чи після.' if body.get('json') else ''
        try:
            chunks = open_upstream(messages, system, body.get('tier', 'default'))
        except AIError as e:
            print('  ШІ: ' + e.message)
            return self._json(e.status, {'code': e.code, 'message': e.message})
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
        try:
            for piece in chunks:
                if piece:
                    self.wfile.write(piece.encode('utf-8'))
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # користувач натиснув «Зупинити»
        except Exception as e:
            print('  ШІ перервався: %s' % e)
        self.close_connection = True

    def _search(self):
        mode = search_mode()
        if mode == 'none':
            return self._json(501, {'code': 'no_search', 'message': 'Пошук в інтернеті вимкнено: впиши OLLAMA_API_KEY (безкоштовно) або ключ Claude API у ' + SETTINGS_FILE + '.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 60_000:
                raise ValueError('size')
            b = json.loads(self.rfile.read(length).decode('utf-8'))
            task = {'context': _clip(b.get('context'), 400), 'budget': _clip(b.get('budget'), 40), 'city': _clip(b.get('city'), 60),
                    'kinds': [k for k in b.get('kinds', []) if k in KIND_WORDS], 'today': _clip(b.get('today'), 40),
                    'profile': b.get('profile') if isinstance(b.get('profile'), dict) else {},
                    'exclude': [_clip(x, 120) for x in b.get('exclude', []) if isinstance(x, str)][:20]}
        except Exception:
            return self._json(400, {'code': 'invalid_request', 'message': 'Неправильний запит.'})
        try:
            print('  Пошук (%s): %s' % (mode, task['context'] or 'за профілем'))
            out = claude_search(task) if mode == 'claude' else ollama_search(task)
        except AIError as e:
            print('  Пошук: ' + e.message)
            return self._json(e.status, {'code': e.code, 'message': e.message})
        except Exception as e:
            print('  Пошук зламався: %s' % e)
            return self._json(502, {'code': 'upstream_error', 'message': 'Пошук не вдався. Спробуй ще раз.'})
        return self._json(200, out)

    def log_message(self, fmt, *args):
        if args and str(args[1] if len(args) > 1 else '').startswith(('4', '5')):
            sys.stderr.write('  %s %s\n' % (self.command, fmt % args))


def main():
    port = CFG['port']
    server = None
    for p in range(port, port + 10):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', p), Handler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        print('Не вдалося зайняти порт %d–%d. Закрий інші копії сервера.' % (CFG['port'], CFG['port'] + 9))
        sys.exit(1)
    url = 'http://127.0.0.1:%d/' % port
    names = {'anthropic': 'Claude API', 'ollama': 'Ollama на цьому комп’ютері', 'openai': 'OpenAI-сумісна модель (%s)' % CFG.get('base', ''), 'none': 'не знайдено'}
    print('=' * 60)
    print('  Поради бабусі запущено: ' + url)
    print('  ШІ: ' + names.get(CFG['provider'], CFG['provider']) + (', модель ' + CFG['model'] if CFG.get('model') else ''))
    if not CFG['configured']:
        print('  ' + CFG.get('hint', ''))
    elif CFG.get('all_models') and len(CFG['all_models']) > 1:
        print('  Інші моделі Ollama: ' + ', '.join(m for m in CFG['all_models'] if m != CFG['model']) + ' (обрати: OLLAMA_MODEL у ' + SETTINGS_FILE + ')')
    sm = search_mode()
    print('  Пошук в інтернеті: ' + {'claude': 'Claude з вебпошуком', 'ollama': 'вебпошук Ollama', 'none': 'вимкнено (див. ' + SETTINGS_FILE + ')'}[sm])
    print('  Зупинити: закрий це вікно або натисни Ctrl+C')
    print('=' * 60)
    if '--no-browser' not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nСервер зупинено.')


if __name__ == '__main__':
    main()

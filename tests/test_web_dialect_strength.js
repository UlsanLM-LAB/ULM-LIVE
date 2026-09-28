// Run with: node --test tests/test_web_dialect_strength.js
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(join(__dirname, '../web/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(saved = null, unavailable = false) {
  const elements = new Map();
  function element() {
    return { textContent: '', innerHTML: '', setAttribute() {}, appendChild() {}, querySelector() { return element(); } };
  }
  const options = [0, 1, 2, 3].map(value => ({
    value: String(value), checked: false,
    addEventListener(event, handler) { this[event] = handler; },
  }));
  const storage = {
    saved,
    getItem(key) { assert.equal(key, 'ulm_dialect_strength'); if (unavailable) throw Error('blocked'); return this.saved; },
    setItem(key, value) { assert.equal(key, 'ulm_dialect_strength'); if (unavailable) throw Error('blocked'); this.saved = value; },
  };
  const context = vm.createContext({
    document: {
      body: { dataset: {} },
      createElement() { return element(); },
      querySelectorAll() { return options; },
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
    },
    localStorage: storage,
    Blob, AbortController, TextDecoder,
    URL: { createObjectURL() { return 'blob:mock'; }, revokeObjectURL() {} },
    Audio: class { play() { return Promise.resolve(); } pause() {} },
    fetch: async () => ({ json: async () => ({ status: 'ready' }) }),
  });
  vm.runInContext(script, context);
  return { context, options, storage, elements };
}

function payload(context, history = []) {
  return JSON.parse(JSON.stringify(context.buildChatSpeechPayload('오늘 뭐하노?', history)));
}

for (const strength of [0, 1, 2, 3]) {
  test(`restores saved strength ${strength} and includes it in request`, () => {
    const { context, options } = page(String(strength));
    assert.equal(payload(context).dialect_strength, strength);
    assert.deepEqual(options.filter(o => o.checked).map(o => Number(o.value)), [strength]);
  });
}

for (const saved of [null, '', '-1', '4', 'strong', 'null', '2.0', ' 2', '1e0']) {
  test(`invalid/missing localStorage ${JSON.stringify(saved)} falls back to 2`, () => {
    assert.equal(payload(page(saved).context).dialect_strength, 2);
  });
}

test('storage errors preserve default and allow in-memory selection', () => {
  const { context, options } = page(null, true);
  assert.equal(payload(context).dialect_strength, 2);
  context.setDialectStrength(0);
  assert.equal(payload(context).dialect_strength, 0);
  assert.ok(options[0].checked);
});

test('radio changes affect the next payload, persist across reload and preserve history', () => {
  const { context, options, storage, elements } = page();
  const history = [{ role: 'user', content: '기억해 줘' }, { role: 'assistant', content: '알겠습니다' }];
  const original = JSON.stringify(history);
  options[3].checked = true;
  options[3].change();
  assert.equal(payload(context, history).dialect_strength, 3);
  assert.deepEqual(payload(context, history).history, history);
  assert.equal(JSON.stringify(history), original);
  assert.equal(storage.saved, '3');
  assert.equal(payload(page(storage.saved).context).dialect_strength, 3);
  options[0].checked = true;
  options[0].change();
  assert.equal(payload(context, history).dialect_strength, 0);
  assert.equal(options.filter(o => o.checked).length, 1);
  elements.get('clear').onclick();
  assert.equal(payload(context).dialect_strength, 0);
});

test('native radio controls have accessible descriptions', () => {
  assert.equal((html.match(/<input[^>]*name="dialect-strength"/g) || []).length, 4);
  assert.equal((html.match(/aria-label="(?:표준어|약하게|보통|강하게):/g) || []).length, 4);
  assert.match(html, /<legend>사투리 강도<\/legend>/);
});


test('microphone completion sends the current strength and clean multi-turn history', async () => {
  const { context } = page();
  const requests = [];
  const initialHistory = [{ role: 'user', content: '첫 인사' }, { role: 'assistant', content: '안녕하세요' }];
  vm.runInContext('history=' + JSON.stringify(initialHistory), context);
  context.blobToWav = async blob => blob; // Audio conversion is outside style control.
  context.fetch = async (url, options) => {
    if (url === '/v1/audio/transcriptions') return { ok: true, json: async () => ({ text: '새 발화' }) };
    assert.equal(url, '/v1/chat/speech');
    requests.push(JSON.parse(options.body));
    return {
      ok: true,
      headers: { get() { return Buffer.from('답변').toString('hex'); } },
      blob: async () => new Blob(['mock wav']),
    };
  };
  for (const strength of [3, 0]) {
    context.setDialectStrength(strength);
    vm.runInContext("chunks=[];recorder={mimeType:'audio/webm',stop(){this.onstop();}}", context);
    await context.stopListening();
  }
  assert.deepEqual(requests.map(request => request.dialect_strength), [3, 0]);
  assert.equal(requests[0].prompt, '새 발화');
  assert.deepEqual(requests[0].history, initialHistory);
  assert.deepEqual(requests[1].history, initialHistory.concat([
    { role: 'user', content: '새 발화' }, { role: 'assistant', content: '답변' },
  ]));
  assert.ok(requests.every(request => request.history.every(message => message.role !== 'system')));
});

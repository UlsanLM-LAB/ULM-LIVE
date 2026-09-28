# Dialect strength (2026-09-28)
- [x] Trace speech request, text payload and browser state without changing TTS settings.
- [x] Validate and propagate dialect_strength to text backend; default2, integer0–3.
- [x] Add compact accessible selector, persisted state and request-time payload inclusion.
- [x] Test mocked speech propagation, JavaScript fallback/payload, desktop/mobile UI and CPU regression suite.
- [ ] Document prompt-based control; commit/push feature branch and create PR without merge; AWS unchanged.

## Review

CPU/offline full pytest: 103 passed, 4 existing integration tests deselected; focused speech API tests: 18 passed. Node web tests: 17 passed, including two mocked microphone turns with different strengths and preserved history. Chromium smoke: default2, select3, persist/reload, invalid-storage fallback2, native keyboard selection and focus outline; 320px viewport has no horizontal overflow. TTS parameters/adapter/voice configuration are unchanged. Qwen-TTS is imported only when the real runtime is constructed so API mocks run without it. AWS and models were not started; actual voice/model style A/B remains untested.

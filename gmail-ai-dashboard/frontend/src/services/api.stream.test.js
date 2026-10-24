import { ReadableStream } from 'node:stream/web';
import { TextDecoder, TextEncoder } from 'node:util';
import { afterEach, expect, it, vi } from 'vitest';
import { sendRagChat } from './api';

afterEach(() => vi.unstubAllGlobals());

function streamResponse(text) {
    const bytes = new TextEncoder().encode(text);
    return { ok: true, body: new ReadableStream({ start(controller) {
        for (let i = 0; i < bytes.length; i += 5) controller.enqueue(bytes.slice(i, i + 5));
        controller.close();
    } }) };
}

it('decodes activity and a final answer across split Unicode stream chunks', async () => {
    vi.stubGlobal('TextDecoder', TextDecoder);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse(
        ': keepalive\n\nevent: activity\ndata: {"action":"retrieve","observation":"Read Café email"}\n\n' +
        'event: result\ndata: {"answer":"Café interview","citations":[]}\n\n'
    )));
    const progress = vi.fn();
    const result = await sendRagChat({ query: 'Find mail' }, progress);
    expect(progress).toHaveBeenCalledWith(expect.objectContaining({ observation: 'Read Café email' }));
    expect(result.data.answer).toBe('Café interview');
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/chat/stream'), expect.objectContaining({ credentials: 'include' }));
});

it('reports stream failure without replaying a potentially mutating request', async () => {
    vi.stubGlobal('TextDecoder', TextDecoder);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse('event: error\ndata: {"message":"Request failed"}\n\n')));
    await expect(sendRagChat({ query: 'Draft a reply' }, vi.fn())).rejects.toThrow('Request failed');
    expect(fetch).toHaveBeenCalledTimes(1);
});

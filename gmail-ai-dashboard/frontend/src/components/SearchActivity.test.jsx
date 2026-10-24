import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import SearchActivity from './SearchActivity';

it('shows the latest actual stage and a private LangSmith trace link', () => {
    render(<SearchActivity steps={[
        { action: 'retrieve', observation: 'Searching mail', status: 'running', elapsed_ms: 10 },
        { action: 'retrieve', observation: 'Found 12 candidates', status: 'completed', elapsed_ms: 2200 },
    ]} tracingEnabled traceUrl="https://smith.langchain.com/o/workspace/projects/p/project/r/run" />);
    expect(screen.getByText('Found 12 candidates')).toBeInTheDocument();
    expect(screen.queryByText('Searching mail')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Open LangSmith trace/ })).toHaveAttribute('rel', 'noopener noreferrer');
    expect(screen.getByText('2.2s')).toBeInTheDocument();
});

it('does not render an untrusted trace URL and explains tracing is off', () => {
    render(<SearchActivity steps={[{ action: 'route', observation: 'Mailbox request selected' }]} traceUrl="javascript:alert(1)" />);
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText(/LangSmith tracing is off/)).toBeInTheDocument();
});

// server.js
import express from 'express';
import { createRequire } from 'module';
import path from 'path';
import { fileURLToPath } from 'url';
import ollama from 'ollama';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static('public'));

// Health check endpoint for Ollama
app.get('/api/health', async (req, res) => {
    try {
        await ollama.list();
        res.json({ status: 'ok', message: 'Successfully connected to Ollama' });
    } catch (error) {
        res.status(500).json({ 
            status: 'error', 
            message: 'Could not connect to Ollama. Is it running?',
            error: error.toString()
        });
    }
});

// Serve the HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Handle Ollama requests using the ollama library
app.post('/api/ask-query', async (req, res) => {
    const { query } = req.body;

    try {
        console.log('Attempting to connect to Ollama with query:', query);
        const response = await ollama.chat({
            model: 'llama3.2',  // 3.2B parameter model
            messages: [{ role: 'user', content: query }],
        });

        console.log('Got response from Ollama:', response);
        res.json({ reply: response.message.content });
    } catch (error) {
        console.error('Detailed error:', error);
        res.status(500).json({ 
            error: `Failed to get response from Ollama: ${error.message}`,
            details: error.toString()
        });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
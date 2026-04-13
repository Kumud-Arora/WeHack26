# Your AI Financial Coach

Upload your bank statement, call a phone number, and ask your AI anything about your money, or if you think buying a tesla on a random tuesday is fine...

## Demo

[![Watch the demo](https://img.youtube.com/vi/xCoycVk6AC8/0.jpg)](https://youtu.be/xCoycVk6AC8)

---

## What it does

You upload a PDF of your bank statement. FinBot reads it, figures out where your money went, and stores that info. Then you can call a Twilio phone number and just... talk to it. Ask things like:

- *"How much did I spend on food last month?"*
- *"What's my biggest expense category?"*
- *"Do you think I can afford a new iPhone?"*

It answers based on your **actual statement data** — not generic advice. After each call it also quietly analyzes how you talked about money (hesitation, impulse buying, decision speed) and uses that to personalize future conversations.

---

## How the pieces fit together

```
Upload PDF
    ↓
Parser extracts transactions, balance, due dates, categories
    ↓
You link your phone number to your statement on the results page
    ↓
You call the Twilio number
    ↓
Gemini AI answers your questions using your real financial data
    ↓
Call ends → behavioral analysis runs in the background
    ↓
Next call, AI already knows your spending patterns and coaching style
```

---

## Stack

- **Flask** — web server + webhook handler
- **pdfplumber + PyMuPDF** — PDF parsing
- **Gemini 2.5 Flash** — AI responses during calls
- **Twilio** — phone call handling (speech-to-text + text-to-speech)
- **Cloudflare Tunnel** — exposes your local Flask server to Twilio
- **JSON files** — stores profiles, statements, and behavioral memory (no database)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create your `.env` file

Copy `.env.example` to `.env` and fill in your keys:

```env
FLASK_SECRET_KEY=anything-random
FLASK_PORT=5000
FLASK_DEBUG=true

GEMINI_API_KEY=your_key_from_aistudio.google.com
GEMINI_MODEL=gemini-2.5-flash

TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=+1234567890

BASE_URL=https://your-cloudflare-tunnel-url.trycloudflare.com
```

### 3. Run the app

```bash
python app.py
```

Go to `http://localhost:5000`

### 4. Expose it to Twilio (for voice calls)

```bash
cloudflared.exe tunnel --url http://localhost:5000
```

Copy the `https://*.trycloudflare.com` URL it gives you, paste it into your `.env` as `BASE_URL`, and set your Twilio phone number's webhook to:

```
https://your-url.trycloudflare.com/voice/incoming
```

---

## Using it

1. Go to `http://localhost:5000` and upload a bank statement PDF
2. On the results page, scroll down and enter your phone number to link it to your statement
3. Call your Twilio number
4. Ask it anything about your finances

---

## What the AI knows about you

Every time you finish a call, FinBot runs a quick analysis on the conversation and tracks things like:

- How often you hesitate before a financial decision
- Whether you tend toward impulse spending
- What topics you bring up most (dining, shopping, bills, etc.)
- Which coaching style ("friendly", "direct", etc.) works best for you

This builds up over time in `call_analyzer/data/memory.json` and shapes how future calls go.

---

## Project structure

```
app.py                      ← Flask app, main routes
parser.py                   ← PDF → JSON transaction extractor
categorizer.py              ← Labels transactions by category
voice/
  routes.py                 ← Twilio webhooks (incoming call, speech, status)
  context_builder.py        ← Assembles what Gemini knows before each response
  gemini_client.py          ← Sends prompts to Gemini, cleans up the reply
  session.py                ← Tracks active call sessions in memory
call_analyzer/
  main/postprocess.py       ← Runs after each call ends
  services/
    behavior_extractor.py   ← Reads transcript, finds behavioral signals
    metrics_calculator.py   ← Turns signals into scores
    memory_store.py         ← Saves + updates rolling behavioral memory
    summarizer.py           ← Writes a plain-English summary of the call
data/profiles.json          ← Phone number → statement + call history
outputs/                    ← Parsed statement JSONs
```

---

## Built at WeHack 2026

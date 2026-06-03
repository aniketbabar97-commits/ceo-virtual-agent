# CEO Virtual Agent - Autonomous Money-Making AI System
**Thinks like a CEO. Hires specialized agents. Researches & scrapes the internet ethically. Learns from experience. Aims to build real revenue streams autonomously (with your oversight for real actions).**

Built for **non-coders**. Fully web-based dashboard (Streamlit). Cloud-deployable. Switch from expensive OpenAI to **free/cheap open-source models** via Groq (fast, free tier) or **completely free local Ollama** (unlimited, no tokens worry, no API keys).

**Date**: 2026-06-03

## ⚠️ IMPORTANT WARNINGS (Read First!)
- **This is experimental/prototype**. Making "real money" autonomously is **very hard**, risky, and not guaranteed. Most attempts fail or take time/capital.
- **Not financial, legal, or investment advice**. You are responsible for all actions, earnings, taxes, compliance.
- **Risks**: 
  - Agent may suggest ideas that lose money (e.g. bad trades, bad products).
  - Scraping: Must be ethical/legal. Agent is instructed to respect robots.txt, rate limits, only public data. **Do not use for spam, fraud, or violating ToS**.
  - Accounts: For real monetization (affiliates, ads, sales), you need to set up accounts (Stripe, Gumroad, Google Ads, etc.). Agent can generate content/links but **human approval needed for spending or posting**.
  - API costs: Groq free tier has limits (~1000+ requests/day). Local Ollama is free/unlimited but needs good computer.
  - Autonomous loops: Can run forever; monitor, set limits.
- Start small: Use for **research and idea validation** first. Approve every revenue-generating step.
- Test locally before cloud.
- If agent suggests illegal/unethical: It should refuse (built-in safeguards).
- Revenue tracking: Starts simulated. Input real earnings manually for learning.

**Goal of this system**: A self-improving "CEO brain" that runs 24/7 in cloud, hires "employees" (AI agents), gathers intel from web, executes digital tasks, learns what works for profit, and builds sustainable income (e.g. content sites, digital products, automated services).

## How to Get Started (Non-Coder Friendly - Step by Step)

### Step 1: Install Prerequisites (One-time, ~10-30 mins)
1. **Install Python 3.10+**:
   - Windows: Download from python.org (check "Add to PATH").
   - Mac: `brew install python` or from python.org.
   - Linux: `sudo apt install python3 python3-pip`

2. **Install Git** (for cloud later):
   - Windows/Mac: Download Git.
   - Verify: Open terminal/command prompt, type `git --version`

3. **Choose your AI Brain (Critical - replaces OpenAI)**:
   - **Option A: Groq (Recommended for beginners - Fast, Free tier, Open models like Llama)**:
     - Go to https://groq.com
     - Sign up (free, often no credit card for tier).
     - Go to API Keys, create one. Copy it.
     - Free limits are generous for starting (thousands of requests/day on good models).
   - **Option B: Ollama (Completely FREE, local, open-source, unlimited - No tokens, no keys, no internet for AI calls)**:
     - Download from https://ollama.com
     - Install and run.
     - In terminal: `ollama pull llama3.2:3b` (small, fast start) or better `ollama pull qwen2.5:7b` or `llama3.1:8b` (needs ~8-16GB RAM).
     - For better quality (if you have strong PC/GPU/Mac M1+ 16GB+): `ollama pull llama3.3:70b` or latest Qwen/Llama 70B (check hardware: 70B needs 40GB+ RAM or quantized).
     - Test: `ollama run llama3.2:3b` then ask something, `/bye` to exit.
     - Runs on your computer - agent won't "stop" ever due to tokens.

4. **Download this project**:
   - Later for cloud, but first: You will get files from me (or clone if on GitHub).

### Step 2: Set Up the Project Locally
1. Open terminal/command prompt.
2. Create folder: `mkdir ceo-agent && cd ceo-agent`
3. Copy all files from this workspace into it (I will help provide them; use copy-paste or zip if possible).
4. Install packages:
   ```
   pip install -r requirements.txt
   ```
   (May take 5-10 mins. If errors, update pip: `pip install --upgrade pip`)

5. (Optional) Create `.env` file for secrets (edit with notepad):
   ```
   GROQ_API_KEY=your_groq_key_here
   ```

### Step 3: Run the Web Dashboard (Your Control Center - Accessible on this computer)
```
streamlit run app.py
```
- Opens in browser at http://localhost:8501
- **Access anywhere on your network**: Use your computer's IP (e.g. http://192.168.x.x:8501)
- Configure in sidebar: Choose Groq or Ollama, paste key or leave for Ollama.
- Set your **Big Goal** (e.g. "Build a profitable AI newsletter or affiliate site in the fitness niche to generate $1000/month passive income within 6 months")
- Click buttons to run "One CEO Cycle" (plans, hires agents, researches/scrapes, executes digital tasks, learns).
- Watch logs, hired agents, learnings, simulated + real revenue.
- "Autonomous Mode": Run multiple cycles automatically (with sleeps).
- Everything saved in files (goal.txt, memory.json, revenue_log.json, agent_logs.txt) - persists.

**First Run Tip**: Start with small goal like "Research and validate 3 profitable side hustle ideas in 2026 with low capital". Run 1-2 cycles. Review outputs.

### Step 4: Deploy to GitHub + Render.com (Cloud Access Anywhere - Recommended, Just Like the Example You Shared)

**Yes — we can (and should) do exactly GitHub + Render**, just like the live demo you linked (https://ceo-agent-etct.onrender.com/).

**Benefits**:
- Your CEO agent runs in the cloud 24/7 (dashboard accessible from phone, laptop, anywhere).
- Changes: You tell me what to improve ("add better YouTube video features", "make the UI look more like that example", "add real upload to YouTube"). I edit the files here. You push to GitHub → one-click redeploy on Render.
- Same platform as the example you showed.

#### Non-Coder Step-by-Step (GitHub + Render Deploy)

1. **Prepare your local folder** (do this once):
   - Make sure you have the 4 main files in a folder called `ceo-virtual-agent`:
     - `app.py`
     - `ceo_core.py`
     - `requirements.txt`
     - `README.md`
     - (Bonus files we added: `render.yaml`, `.gitignore`)
   - Also create empty `data/` and `generated/` folders.

2. **Create a GitHub Repository**:
   - Go to https://github.com → New repository.
   - Name it `ceo-virtual-agent` (public is fine to start).
   - **Do NOT** initialize with README (we already have one).
   - Click "Create repository".
   - On the next screen, copy the commands under "…or push an existing repository from the command line".

3. **Push the code to GitHub** (in your terminal, inside the `ceo-virtual-agent` folder):
   ```bash
   git init
   git add .
   git commit -m "Initial CEO Virtual Agent"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ceo-virtual-agent.git
   git push -u origin main
   ```
   (Replace YOUR_USERNAME with your GitHub username. If it asks for login, use GitHub token or CLI.)

4. **Deploy on Render.com** (free tier available):
   - Go to https://render.com → Sign up (free, GitHub login is easiest).
   - Click **"New +"** → **Web Service**.
   - Connect your GitHub account and select the `ceo-virtual-agent` repo.
   - Settings:
     - **Name**: `ceo-virtual-agent` (or anything)
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
     - **Plan**: Free (or Starter ~$7/mo for no sleep + persistent disk)
   - Click "Advanced" → **Environment Variables** and add:
     - `GROQ_API_KEY` = your Groq key (from groq.com)
     - (Optional) `LLM_PROVIDER` = `groq`
   - Click "Create Web Service".

5. **Your live URL**:
   - Render will give you something like `https://ceo-virtual-agent-xxx.onrender.com`
   - Open it — this is your cloud dashboard, accessible anywhere!

6. **For real 24/7 autonomous runs** (important):
   - Free Render web services sleep after ~15 minutes of no traffic.
   - Options:
     - Upgrade to paid plan (no sleep).
     - Or: Deploy the **dashboard** as Web Service (for UI + approvals).
     - Deploy a separate **Background Worker** on Render running `python ceo_core.py --autonomous --cycles 1000` (or set it to loop forever).
     - The dashboard will still show the generated content and approvals when you visit.

**To update later (super easy)**:
- Tell me the change.
- I update the files in this workspace.
- You do `git pull` (or just commit the new files) and push to GitHub.
- Render auto-deploys in ~1-2 minutes.

We also included a `render.yaml` file for even easier future deploys (Infrastructure as Code).

**Secrets note**: Never commit your Groq key. Always set it in Render Environment Variables (or use Streamlit secrets if deploying to Streamlit Cloud instead).

This setup is exactly what the example you linked uses — but our version has the Owner approval system, real video/voice generation for YouTube, open models, and the full CEO hiring + learning loop you wanted.

Would you like me to:
- Add more Render-specific tweaks (e.g. better logging for background worker)?
- Create a separate `worker.py` or instructions for the autonomous background part?
- Make the UI closer to the example you showed (more metric cards, "Build Team" button, etc.)?

Just say the word and I'll prepare the exact files + updated instructions. Then you can deploy your own version in 10-15 minutes. 

Ready when you are! 🚀

(If you want, paste the live URL of your deployed version later and I can help customize it further.)

## How the CEO Agent Works — Completely Autonomous with Owner Oversight (You as Owner)

The system is now designed exactly as you requested: **completely autonomous** for most work, but it **asks you (the Owner)** for verification on high-stakes decisions and executions.

**Your Role**: "Owner" — the human boss. The CEO reports to you and must get your explicit approval for anything that creates accounts, posts publicly, uploads videos, spends money, or uses external APIs.

**Autonomous Capabilities (the agent does these without asking every time)**:
- Research niches, competitors, trends (searches + ethical scraping).
- Hire and delegate to specialist agents (researcher, scraper, content creator, analyst, executor).
- Generate high-quality content, scripts, SEO assets, landing pages (saved to generated/).
- **Real execution tools**:
  - Generate voiceovers (real MP3 using free gTTS) for videos/podcasts.
  - Create real faceless YouTube-style videos (MP4 slideshow + text overlays + optional voiceover using moviepy).
  - Prepare full YouTube upload packages (video + metadata.json + thumbnail).
  - Log revenue and learn from what actually makes money.
- Run in loops autonomously (plan → execute → reflect → learn → sleep → repeat).
- "Arena" style decision making for best strategies.
- Self-improves via persistent memory of lessons.

**When it asks the Owner (mandatory pause)**:
- Setting up a new YouTube channel or social account.
- Uploading/posting a video or content publicly.
- Integrating or using paid APIs / spending money.
- Any action with significant risk or external account creation.

**Flow Example (YouTube channel as you described)**:
1. CEO researches "best faceless YouTube niches 2026" + scrapes competitors.
2. Decides "Best idea: Productivity tips for solopreneurs. Create channel 'AI Daily Edge'".
3. Uses `request_owner_approval` tool with full details (rationale, projected revenue, risks, what it needs from you).
4. **Pauses**. The web dashboard shows a clear "Pending Owner Approval" card with all info.
5. **You (Owner)** review, type your inputs (e.g. "Approved. Channel name: AI Productivity Daily. Use voice + slideshow style. I created the Google account."), click APPROVE.
6. Next cycle: CEO sees the approval + your details in context, generates script, voiceover (real MP3), creates real MP4 video file, prepares upload package.
7. If needed, requests approval again for the actual upload (you provide YouTube API credentials once for full autonomy).
8. Agent can then analyze performance (research or via API) and iterate.

This gives you full control as Owner while the team runs 24/7 on everything else.

See the dashboard "👑 Owner Approvals" section for pending requests. The agent will only proceed on approved actions.

## How the CEO Agent Works (Thinks like a CEO)
- **CEO Mindset Prompt**: Strategic, profit-focused, ROI-obsessed, risk-aware, long-term builder, team leader ("hires" by delegating), data-driven, pivots fast, builds moats (audience, IP, systems).
- **Hires Other Agents**: Uses hierarchical CrewAI - CEO (manager) dynamically plans and delegates to specialized "employees":
  - Market Researcher (searches trends, competitors, niches).
  - Web Scraper/Summarizer (deep dives pages ethically).
  - Content & Product Creator (blogs, emails, landing pages, digital goods, code).
  - Financial Analyst (projections, break-even, scaling math).
  - Executor & Growth Hacker (SEO, social strategy, launch plans, asset generation).
- **Scrapes Internet**: Custom tools for search (DuckDuckGo - free, no key) + page scraping (BeautifulSoup). Ethical instructions built-in.
- **Makes Money**: 
  - Researches real opportunities (niches with demand + low competition + monetizable: affiliates, ads, products, services).
  - Generates ready-to-use assets (articles ready for WordPress/Medium, HTML sites, email sequences, product ideas with descriptions).
  - Proposes execution plans with estimated revenue/costs/timeline.
  - "Virtual revenue" simulated from research; you log **real earnings** (e.g. "Sold product for $47 from agent's Gumroad idea").
  - Over time: Learns what converts (e.g. "Fitness email sequences work better than X").
- **Completely Autonomous Loop**:
  1. CEO reviews goal + memory + current "business state".
  2. Creates daily/weekly plan.
  3. Hires/delegates to crew (or "arena": multiple ideas compete).
  4. Crew executes (search, scrape, create, analyze).
  5. CEO reviews output, decides next (or "hire more", pivot).
  6. Reflects: Extracts lessons, updates memory (what worked for profit?).
  7. Logs everything, updates revenue projections.
  8. Sleeps (configurable, e.g. 1hr or daily).
- **Learns from it**: Persistent memory.json + logs. CEO prompts include "Past lessons: ...". After cycles, "reflection task" improves strategy. Self-improving over time.
- **Arena-style**: For big decisions, CEO can spawn "proposal arena" - 2-3 agents propose competing strategies, CEO picks winner based on CEO criteria (profit potential, risk, speed, scalability).

## Files in This Project
- `app.py`: The web dashboard (your control panel).
- `ceo_core.py`: Core agent logic, tools, memory, autonomous runner (can run standalone for cloud server).
- `requirements.txt`: All Python deps.
- `README.md`: This guide.
- Data files created on run: `data/goal.txt`, `data/memory.json`, `data/revenue.json`, `data/logs.txt`, `generated/` for outputs.

## Next Steps After Setup
1. Run locally, set a goal, run 3-5 cycles manually. Review every output in dashboard.
2. Input some "real" (or test) revenue to bootstrap learning.
3. Deploy to cloud as above.
4. Tell me improvements: "Make the CEO more aggressive on revenue" or "Add Twitter/X posting tool" or "Focus on SaaS idea generation" or "Add real Stripe integration simulation".
5. Scale: Once profitable ideas validated, add real integrations (I can help: e.g. email via Resend free tier, etc.).
6. For real execution: Agent generates "deployable" things like static sites you can host free on GitHub Pages/Netlify, or code for bots.

## Troubleshooting (Non-Coder)
- **No internet in agent?** Check firewall. DuckDuckGo tool needs net.
- **Ollama not connecting**: Make sure `ollama serve` running, model pulled, correct model name in UI (e.g. "llama3.2:3b").
- **Groq errors**: Check key, model name (try "llama-3.3-70b-versatile" or check Groq console for available). Free tier may have RPM limits - agent waits/retires.
- **Slow or expensive**: Use smaller model or local. Limit cycles.
- **Agent does nothing or loops bad**: Check logs. Increase "max cycles". CEO prompt has safeguards.
- **Install errors**: `pip install --upgrade pip setuptools wheel` then retry. Or use conda.
- **Streamlit not opening**: Run in terminal where you cd'ed to folder.
- **Changes**: Tell me exactly, I'll update files. Redeploy.

## Advanced / Future (Tell me to add)
- Real integrations: Gumroad API, email, social posting (with approval), payment links.
- Browser automation (Playwright) for more scraping/acting.
- Vector memory (Chroma) for better learning.
- Multi-model arena (different LLMs compete).
- Specific business templates (e.g. "Newsletter CEO", "Ecom Arbitrage CEO").
- Mobile-friendly or Telegram bot control.
- Cost tracking per cycle (tokens used).
- Export reports, charts.

## Support
Tell me in chat:
- "Update the goal logic to be more conservative on risk."
- "Add a new agent role for 'Affiliate Link Researcher'."
- "Help deploy to Railway - give exact steps."
- "Change to focus on building and selling digital products."
- "Make the dashboard show graphs of projected vs real revenue."

**You are the real CEO. This is your tireless, always-learning virtual executive team.** Start small, learn, iterate, and it can compound into real income.

Let's get you set up - reply with questions or "build the files now, I'm ready to run locally first" or specific goal.

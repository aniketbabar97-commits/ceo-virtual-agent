#!/usr/bin/env python3
"""
CEO Virtual Agent - Web Dashboard (Streamlit)
Non-coder friendly control center for the autonomous CEO agent.
Access anywhere once deployed to cloud. 
Tell the AI (me) to make changes - I edit the code, you redeploy.
"""

import streamlit as st
import os
import json
import time
import datetime
from pathlib import Path
import subprocess
import sys

# Import core logic
try:
    from ceo_core import (
        run_one_ceo_cycle, run_autonomous, get_config, save_config,
        load_goal, save_goal, load_memory, load_revenue,
        DATA_DIR, GENERATED_DIR, LOGS_FILE
    )
except ImportError as e:
    st.error(f"Core import error: {e}. Make sure ceo_core.py is in the same folder.")
    st.stop()

# ============== PAGE SETUP ==============
st.set_page_config(
    page_title="CEO Virtual Agent | Autonomous Money Machine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for CEO feel
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1a365d; margin-bottom: 0.5rem;}
    .ceo-quote {font-style: italic; color: #4a5568; border-left: 4px solid #3182ce; padding-left: 1rem;}
    .metric-card {background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 1rem; border-radius: 10px; border: 1px solid #e2e8f0;}
    .success-box {background-color: #c6f6d5; padding: 1rem; border-radius: 8px;}
    .warning-box {background-color: #fef3c7; padding: 1rem; border-radius: 8px;}
    .asset-card {border: 1px solid #cbd5e0; padding: 0.75rem; margin: 0.5rem 0; border-radius: 6px; background: white;}
    .log-line {font-family: monospace; font-size: 0.85rem; background: #f8f9fa; padding: 2px 6px; margin: 2px 0; border-radius: 3px;}
    .stButton>button {background: linear-gradient(90deg, #2b6cb0, #3182ce); color: white; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.title("💼 CEO Virtual Agent")
st.markdown("**Thinks like a CEO. Hires specialist agents. Scrapes the internet ethically. Learns & compounds. Builds real revenue autonomously.**")
st.caption("Web-based • Cloud-ready • Open-source models (Groq/Gemini free tiers, Ollama local, or DEGRADED KEYLESS mode - works with ZERO API keys!) • Zero OpenAI dependency • Made for non-coders • Never stops without keys")

# ============== SIDEBAR: CONFIG & QUICK ACTIONS ==============
with st.sidebar:
    st.header("🧠 AI Brain Configuration")
    st.caption("Replace any expensive API. Use free open models.")
    
    # Support Streamlit Cloud secrets for deployment (add in app settings or secrets.toml)
    secrets_config = {}
    try:
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            secrets_config['api_key'] = st.secrets['GROQ_API_KEY']
            secrets_config['provider'] = st.secrets.get('LLM_PROVIDER', 'groq')
            secrets_config['model'] = st.secrets.get('LLM_MODEL', 'llama-3.3-70b-versatile')
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            secrets_config['api_key'] = st.secrets['GOOGLE_API_KEY']
            secrets_config['provider'] = 'gemini'
        if hasattr(st, 'secrets') and 'OLLAMA_BASE_URL' in st.secrets:
            secrets_config['ollama_base_url'] = st.secrets['OLLAMA_BASE_URL']
    except Exception:
        pass
    
    config = get_config()
    # Override with secrets if present (for cloud deploys)
    if secrets_config:
        config.update(secrets_config)
        st.caption("Using secrets from cloud deployment (secure).")
    
    # Expanded providers: now includes Gemini (best free tier 2026), degraded keyless, etc.
    provider_options = ["groq", "gemini", "together_ai", "ollama", "degraded (keyless - zero cost, always works)"]
    current_prov = config.get("provider", "groq")
    if current_prov == "degraded":
        current_prov = "degraded (keyless - zero cost, always works)"
    try:
        default_idx = provider_options.index(current_prov)
    except:
        default_idx = 0
    
    provider = st.selectbox(
        "LLM Provider (Open Models + Keyless Fallback)",
        provider_options,
        index=default_idx,
        help="""Groq: Fastest free tier (Llama). 
Gemini: Google's BEST free tier 2026 (Gemini Flash, generous limits, no card often - RECOMMENDED if no key yet).
Together AI / others: Good open models, may need free trial key.
Ollama: 100% local free unlimited (run on your PC).
DEGRADED (keyless): NO API key needed ever! Uses free DuckDuckGo + local gTTS/PIL for research, voice, videos, approvals. CEO still runs, generates assets, learns basics. Add key later for full AI brain. NEVER STOPS without keys."""
    )
    
    # Normalize provider name
    if "degraded" in provider.lower():
        provider = "degraded"
        api_key = ""
        model = "keyless"
        ollama_url = ""
        st.info("✅ DEGRADED KEYLESS MODE selected. System will run fully without any API keys or LLM. Perfect if you have no keys. Real assets + research + approvals still generated!")
    elif provider == "gemini":
        api_key = st.text_input(
            "Google/Gemini API Key (FREE at aistudio.google.com/app/apikey)",
            value=config.get("api_key", ""),
            type="password",
            help="Sign up FREE with Google (no credit card for free tier). Go to Google AI Studio → Get API key. Paste here. Very generous daily limits for Flash models. Best free option for full intelligence."
        )
        model = "gemini-1.5-flash"
        ollama_url = ""
    elif provider == "groq":
        api_key = st.text_input(
            "Groq API Key (free at groq.com)",
            value=config.get("api_key", ""),
            type="password",
            help="Sign up free → Console → API Keys. Paste here. Limits are high enough to start (thousands of calls/day)."
        )
        model = st.selectbox(
            "Model (open-source)",
            ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
            index=0,
            help="70B models are smartest for CEO strategy. Start with versatile."
        )
        ollama_url = ""
    elif provider == "together_ai":
        api_key = st.text_input(
            "Together AI API Key (free trial at together.ai)",
            value=config.get("api_key", ""),
            type="password",
            help="Sign up for free credits/trial. Good for Llama etc. Use Gemini free if you want zero cost."
        )
        model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        ollama_url = ""
    else:  # ollama
        api_key = ""
        model = st.text_input(
            "Ollama Model Name",
            value=config.get("model", "llama3.2:3b"),
            help="Must have Ollama installed & running + model pulled (e.g. llama3.2:3b for fast start, or qwen2.5:7b / llama3.1:8b for better reasoning). See README."
        )
        ollama_url = st.text_input(
            "Ollama Base URL",
            value=config.get("ollama_base_url", "http://localhost:11434"),
            help="Default for local. Change if running Ollama on another machine/server."
        )
    
    interval = st.slider(
        "Autonomous Interval (minutes between cycles)",
        15, 240, config.get("autonomous_interval_minutes", 60), 15,
        help="How often the agent 'wakes up' to think & act when running autonomously."
    )
    
    max_cycles = st.number_input(
        "Max Cycles per Autonomous Run",
        1, 20, config.get("max_cycles_per_run", 5),
        help="Safety limit for 'Run Autonomous' button."
    )
    
    degraded_force = st.checkbox(
        "Force Degraded Keyless Mode (ignore keys, always use zero-cost mode)",
        value=config.get("degraded_mode", False),
        help="Check this to force keyless mode even if keys are set. Useful for testing or to avoid any token use."
    )
    
    if st.button("💾 Save Configuration", type="primary"):
        new_config = {
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "ollama_base_url": ollama_url,
            "autonomous_interval_minutes": interval,
            "max_cycles_per_run": max_cycles,
            "degraded_mode": degraded_force
        }
        save_config(new_config)
        st.success("Config saved! Changes apply to next cycle. If degraded selected, CEO will run keyless (never stops).")
        st.rerun()
    
    st.divider()
    st.header("🚀 Quick Start")
    if st.button("📖 Open Full README Guide"):
        with open("README.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    
    st.caption("Non-coder tip: Start with Groq + small goal. Run 1-2 manual cycles. Review everything before autonomous.")

# ============== MAIN DASHBOARD ==============
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Your Business Goal (The CEO's North Star)")
    current_goal = load_goal()
    new_goal = st.text_area(
        "Edit your high-level objective here. Be specific about niche, revenue target, timeline, constraints (e.g. low capital).",
        value=current_goal,
        height=100,
        help="Example: 'Build a profitable faceless YouTube channel or newsletter in the personal finance niche for beginners, monetized via affiliates and digital products, targeting $800/month within 5 months with under $100 startup cost.'"
    )
    if st.button("Update Goal", type="secondary"):
        save_goal(new_goal.strip())
        st.success("Goal updated. CEO will use this from next cycle.")
        st.rerun()

with col2:
    st.subheader("💰 Revenue Tracker")
    revenue = load_revenue()
    memory = load_memory()
    
    st.metric("Real Revenue Logged (USD)", f"${revenue.get('total_real_usd', 0):.2f}")
    st.metric("Cycles Run (Learned From)", memory.get("total_cycles", 0))
    
    with st.expander("Log Real Earnings (Critical for Learning!)"):
        amt = st.number_input("Amount Earned (USD)", min_value=0.0, step=1.0, value=0.0)
        src = st.text_input("Source (e.g. 'Gumroad product from agent's landing page idea')")
        notes = st.text_area("Notes / What worked", height=60)
        if st.button("Log This Revenue"):
            if amt > 0 and src:
                # Call the tool function directly for logging
                from ceo_core import log_revenue_event
                result = log_revenue_event(amt, src, notes)
                st.success(result)
                st.rerun()
            else:
                st.warning("Enter amount >0 and a source.")

st.divider()

# ============== CONTROL CENTER ==============
st.header("🎮 Control Center - Run the CEO")

run_col1, run_col2, run_col3 = st.columns(3)

with run_col1:
    if st.button("▶️ Run ONE CEO Cycle", type="primary", use_container_width=True):
        with st.spinner("CEO is thinking, hiring agents, researching, scraping, creating assets, learning... This may take 1-5 minutes depending on model."):
            report = run_one_ceo_cycle(get_config())
            if report.get("success"):
                st.success("Cycle completed successfully!")
                st.session_state["last_report"] = report
            else:
                st.error(f"Cycle failed: {report.get('error', 'Unknown')}")
            st.rerun()

with run_col2:
    cycles_to_run = st.number_input("Cycles to run autonomously", 1, 10, 3, key="auto_cycles")
    if st.button(f"🔄 Run {cycles_to_run} Autonomous Cycles", use_container_width=True):
        st.warning("This will run multiple cycles with sleeps. UI may be unresponsive. For true 24/7, run `python ceo_core.py --autonomous --cycles 10` in a separate terminal or server instead.")
        with st.spinner(f"Running {cycles_to_run} full autonomous cycles... (check terminal/logs for progress)"):
            run_autonomous(get_config(), cycles_to_run)
            st.success(f"Autonomous run of {cycles_to_run} cycles finished. Check logs and generated assets.")
            st.rerun()

with run_col3:
    if st.button("🛑 Reset All Data (Memory, Logs, Revenue)", type="secondary", use_container_width=True):
        if st.checkbox("Confirm reset? This clears learning!"):
            for f in [LOGS_FILE, os.path.join(DATA_DIR, "memory.json"), os.path.join(DATA_DIR, "revenue.json")]:
                if os.path.exists(f):
                    os.remove(f)
            st.success("Data reset. Fresh start for CEO.")
            st.rerun()

st.caption("Tip for cloud 24/7: Deploy dashboard here. Run the agent script (ceo_core.py --autonomous) on a cheap VPS or in background. Dashboard reads the same files for monitoring.")

# ============== LATEST RESULTS ==============
st.header("📊 Latest CEO Activity & Outputs")

if "last_report" in st.session_state and st.session_state["last_report"].get("success"):
    report = st.session_state["last_report"]
    st.markdown(f"**Last Cycle:** {report.get('timestamp', 'N/A')}")
    st.markdown(f"**Assets Generated:** {len(report.get('assets_generated', []))}")
    
    with st.expander("📋 Full CEO Cycle Report (What the CEO decided & learned)", expanded=True):
        st.text_area("Raw Output", report.get("full_output", "No output"), height=300, key="report")
    
    st.info(report.get("next_recommended", ""))

# Always show latest from logs
st.subheader("Recent Agent Logs (Live View)")
if os.path.exists(LOGS_FILE):
    with open(LOGS_FILE, 'r', encoding='utf-8') as f:
        logs = f.readlines()[-30:]  # Last 30 lines
    for line in logs:
        st.markdown(f'<div class="log-line">{line.strip()}</div>', unsafe_allow_html=True)
else:
    st.caption("No logs yet. Run a cycle to see the CEO and hired agents in action.")

# ============== GENERATED ASSETS (Execution Proof) ==============
st.subheader("📁 Generated Assets (Ready-to-Use - This is how it 'makes money')")
gen_path = Path(GENERATED_DIR)
assets = sorted([f for f in gen_path.glob("*") if f.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)[:10]

if assets:
    for asset in assets:
        with st.expander(f"📄 {asset.name} (click to preview)"):
            suffix = asset.suffix.lower()
            try:
                if suffix in ['.gif', '.png', '.jpg', '.jpeg']:
                    st.image(str(asset), use_column_width=True)
                    st.caption("Animated GIF / image — ready to use as thumbnail or video placeholder.")
                elif suffix == '.mp3':
                    st.audio(str(asset))
                    st.caption("Voiceover MP3 — generated with free gTTS. Ready for video or podcast.")
                elif suffix == '.mp4':
                    st.video(str(asset))
                    st.caption("MP4 video — ready for YouTube after Owner approval.")
                elif suffix in ['.html', '.htm']:
                    content = asset.read_text(encoding='utf-8', errors='ignore')[:3000]
                    st.components.v1.html(content, height=400, scrolling=True)
                else:
                    content = asset.read_text(encoding='utf-8', errors='ignore')[:2500]
                    st.code(content, language="markdown" if suffix in ['.md', '.txt'] else None)
                st.caption(f"Full path: {asset} | Size: {asset.stat().st_size} bytes | Deploy this yourself (e.g. copy HTML to free host, post article to blog, use as email copy).")
            except Exception as e:
                st.error(f"Preview error: {e}")
else:
    st.caption("No assets generated yet. Run a cycle - the Executor agent will save blog posts, landing pages (HTML), product descriptions, etc. here for you to use immediately.")

# ============== MEMORY & LEARNING (Self-Improvement) ==============
st.subheader("🧠 CEO Memory & Lessons Learned (How it gets smarter over time)")
mem = load_memory()
if mem.get("lessons"):
    st.caption(f"Total lessons stored: {len(mem['lessons'])} (last 10 shown). These are injected into every future CEO prompt.")
    for i, lesson in enumerate(reversed(mem["lessons"][-10:])):
        with st.container(border=True):
            st.markdown(f"**{lesson.get('timestamp', '')[:16]}**")
            st.write(lesson.get("lesson", ""))
            if lesson.get("context"):
                st.caption(f"Context: {lesson['context'][:200]}...")
else:
    st.caption("No lessons yet. Run cycles - after each, the CEO reflects and stores specific 'what worked for profit' insights.")

# ============== TABS FOR DEEPER INFO ==============
tab1, tab2, tab3 = st.tabs(["📈 Revenue History", "🤖 How the Agents Work (Hiring)", "⚖️ Ethics & Scraping Rules"])

with tab1:
    rev = load_revenue()
    st.write(f"**Total Real Revenue Tracked: ${rev.get('total_real_usd', 0):.2f}**")
    if rev.get("events"):
        for evt in reversed(rev["events"][-8:]):
            st.write(f"- ${evt['amount_usd']:.2f} | {evt['source']} | {evt.get('timestamp', '')[:10]} | {evt.get('notes', '')[:80]}")
    else:
        st.caption("Log real earnings above after you execute agent's suggestions (e.g. you launched the product it created and made a sale). This is how the CEO learns what actually converts to money.")
    st.info("Projected revenue is estimated inside cycle reports by the Analyst agent based on research.")

with tab2:
    st.markdown("""
    **The CEO 'hires' a team using CrewAI hierarchical process:**
    
    - **CEO (Manager)**: Strategic planning, arena-style option comparison (multiple strategies compete internally, best wins), delegates, reviews everything, extracts lessons, decides pivots.
    - **Senior Market Researcher**: Searches (DuckDuckGo), validates niches, competitors, demand.
    - **Ethical Web Intelligence Scraper**: Deep dives public pages only (with rate limits & user-agent).
    - **Content & Digital Product Creator**: Writes blogs, emails, landing pages (HTML), sales copy, ebooks. Saves ready files.
    - **Financial & Growth Analyst**: Math on projections, ROI, what metrics needed for $X revenue.
    - **Growth Executor**: Turns plans into checklists, more assets, deployment instructions.
    
    **Hiring in action**: CEO creates the plan first (with 'arena' for best strategy), then the crew of specialists executes their assigned parts. Everything is logged.
    
    No fixed salaries - pure AI, on-demand.
    """)
    st.caption("This is inspired by real CEO workflows and Agent Arena concepts for better decisions.")

with tab3:
    st.warning("""
    **Built-in Ethical & Legal Safeguards (CEO enforces these strictly):**
    
    - Only public data. Never logins, paywalls, private info, or bypassing restrictions.
    - Respects robots.txt (basic check in scraper).
    - Strong rate limiting (1-2+ seconds between requests).
    - Clear User-Agent identifying as research bot.
    - All suggestions must be legal and ethical. Agent refuses fraud/spam.
    - Revenue actions always flagged: "REQUIRES USER APPROVAL".
    - Content is original; no copyright violation.
    
    **Your responsibility**: Comply with all laws (tax, advertising, data privacy), platform ToS (Google, Amazon Associates, etc.), and review everything before publishing or spending.
    
    If agent ever suggests something shady: It shouldn't - but stop and tell me (the builder) to fix the prompt.
    """)
    st.success("Scraping is for research & competitive intelligence only. The goal is legitimate value creation (great content/products that people pay for).")

# ============== FOOTER / NEXT STEPS ==============
st.divider()

# ============== OWNER APPROVALS SECTION (Your Role as Owner) ==============
st.header("👑 Owner Approvals — Your Role as the Boss")
st.caption("The CEO will autonomously research, plan, generate content/voice/videos, and prepare uploads. For high-stakes actions (channel creation, actual posting, API use, spending), it **must** use the request_owner_approval tool and pause. You review here, approve or reject, and provide details (e.g. channel name, confirmation, credentials). Once approved, the next cycle will see it and execute.")

approval_dir = os.path.join(DATA_DIR, "approvals")
os.makedirs(approval_dir, exist_ok=True)

pending = []
approved_recent = []
for f in sorted(os.listdir(approval_dir), reverse=True):
    if f.endswith(".json"):
        try:
            with open(os.path.join(approval_dir, f), "r") as fh:
                req = json.load(fh)
            if req.get("status") == "PENDING":
                pending.append((f, req))
            elif req.get("status") in ["APPROVED", "REJECTED"]:
                approved_recent.append((f, req))
        except:
            pass

if pending:
    st.warning(f"**{len(pending)} Pending Owner Approval(s)** — The autonomous agent is waiting for you on these.")
    for fname, req in pending:
        with st.expander(f"📋 {req['action'][:80]}... (ID: {req['id']})", expanded=True):
            st.write(f"**Rationale:** {req['rationale']}")
            st.write(f"**Estimated Revenue Impact:** {req['estimated_revenue_impact']}")
            st.write(f"**Risks:** {req['risks']}")
            st.write(f"**Required from you (Owner):** {req['required_owner_inputs']}")
            
            owner_input = st.text_area(f"Your response / details for {req['id']}", 
                                       value="", 
                                       key=f"owner_input_{req['id']}",
                                       help="E.g. 'Approved. Channel name: MyAIProductivity. Use slideshow + voiceover style. Here is the credentials.json path if needed: /path/to/credentials.json'")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(f"✅ APPROVE & Provide Inputs", key=f"approve_{req['id']}"):
                    req["status"] = "APPROVED"
                    req["owner_response"] = owner_input or "Approved by Owner with no additional notes."
                    req["approved_at"] = datetime.datetime.now().isoformat()
                    with open(os.path.join(approval_dir, fname), "w") as fh:
                        json.dump(req, fh, indent=2)
                    st.success("Approved! The CEO will see this in the next cycle and proceed with the action using your inputs.")
                    st.rerun()
            with col_b:
                if st.button(f"❌ REJECT", key=f"reject_{req['id']}"):
                    req["status"] = "REJECTED"
                    req["owner_response"] = owner_input or "Rejected by Owner."
                    with open(os.path.join(approval_dir, fname), "w") as fh:
                        json.dump(req, fh, indent=2)
                    st.info("Rejected. CEO will note this and explore alternatives in next cycle.")
                    st.rerun()
else:
    st.success("No pending approvals right now. The agent can run fully autonomously on low-risk tasks (research, drafting, asset generation). High-impact actions will create requests here.")

if approved_recent:
    with st.expander("Recently handled approvals (last 5)"):
        for fname, req in approved_recent[:5]:
            st.write(f"- **{req['action'][:60]}** | Status: {req['status']} | {req.get('approved_at', req['timestamp'])[:16]}")

st.caption("This is how you stay in control as Owner while the CEO team runs 24/7 autonomously on everything else.")

# ============== FOOTER / NEXT STEPS ==============
st.divider()
st.markdown("""
**Next for you (non-coder):**
1. Set/save a specific goal above (content/YouTube focused by default).
2. Configure Groq key (your preference) or Ollama.
3. Click "Run ONE CEO Cycle" — watch it research, hire agents, generate scripts/voice/videos (if moviepy available), and **request your approval** for channel setup or uploads.
4. When a pending approval appears above, review, type your inputs/confirmation, and click APPROVE.
5. Run another cycle — the CEO will now execute the approved action (e.g. actually create the video file + package, or prepare for upload).
6. For full real YouTube autonomy: See README for one-time Google Cloud + OAuth setup (free tier), then approve the action and provide the credentials file path.
7. Log real revenue when you monetize the outputs.
8. Deploy to Streamlit Cloud for access anywhere + background runs.

**Completely autonomous with Owner oversight exactly as you asked.** The agent will make videos, prepare posts, analyze opportunities, and only ask you (the Owner) for verification on the big moves like channel creation or actual publishing.

Tell me what to improve next!
""")

# Auto-refresh hint for logs
if st.checkbox("Auto-refresh logs every 10s (for monitoring long runs)", value=False):
    time.sleep(10)
    st.rerun()

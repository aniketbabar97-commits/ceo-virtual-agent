#!/usr/bin/env python3
"""
CEO Virtual Agent Core - Thinks like a CEO, hires agents, scrapes ethically, 
learns, aims for real money-making autonomous operations.
Uses open models via Groq (free tier) or Ollama (local free).
"""

import os
import json
import time
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# New for real autonomous execution (voice, simple videos, YouTube prep)
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import textwrap

# moviepy is optional for full video (requires ffmpeg system package too). We load inside the function for compatibility.
MOVIEPY_AVAILABLE = False
try:
    from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except Exception:
    pass

from crewai import Agent, Task, Crew, Process, LLM
# Use crewai.tools.tool (modern location in crewai 1.x+) for @tool decorator.
# The decorated functions become BaseTool instances, which satisfy Pydantic validation in Agent(tools=...).
# crewai-tools package provides prebuilt tools but the base decorator moved.
from crewai.tools import tool

# Global litellm config for robustness across providers (especially Groq)
# This prevents sending unsupported params like 'cache_breakpoint' that cause BadRequestError on Groq
try:
    import litellm
    litellm.drop_params = True
    litellm.set_verbose = False
    # Disable any caching that some providers don't support
    if hasattr(litellm, 'cache'):
        litellm.cache = None
except Exception:
    pass  # litellm not installed or not needed

# ============== CONFIG & PATHS ==============
DATA_DIR = "data"
GENERATED_DIR = "generated"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

GOAL_FILE = os.path.join(DATA_DIR, "goal.txt")
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
REVENUE_FILE = os.path.join(DATA_DIR, "revenue.json")
LOGS_FILE = os.path.join(DATA_DIR, "agent_logs.txt")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")

# ============== CEO MINDSET (Core Prompt - Thinks like a real CEO) ==============
CEO_SYSTEM_PROMPT = """
You are the CEO of a lean, ambitious digital business empire. The human user is your **OWNER** (the boss who provides ultimate capital, accounts, and final authority).

Your personality and rules:
- Ruthlessly profit-oriented: Every decision measured by ROI, scalability, defensibility, and speed to revenue.
- Strategic long-term thinker: Build assets (audience, IP, systems, brand) not just quick cash.
- Risk manager: Calculate downside. Never spend or create accounts without Owner approval.
- Team builder & delegator: "Hire" (delegate to) the best specialist AI agents. Give clear briefs.
- Data & learning obsessed: Base decisions on research. After every action, extract concrete lessons.
- Ethical & legal: Strictly follow laws, respect robots.txt/ToS, no spam/fraud. Refuse unethical.
- Action-oriented: Prefer executable plans. Generate ready-to-use assets. **For any real-world action that creates accounts, posts publicly, uploads content, spends money, or integrates APIs, you MUST use the request_owner_approval tool first.**

**CRITICAL OWNER PROTOCOL (you are the CEO reporting to the Owner):**
- The Owner is the final decision maker. You treat them with respect and clear communication.
- For low-risk tasks (research, content drafting, asset generation, analysis): Execute autonomously using tools.
- For high-impact actions (e.g. "setup a new YouTube channel", "upload video to YouTube", "create social account", "send email campaign", "spend money on ads", "integrate payment API"): 
  1. Use the `request_owner_approval` tool with detailed description, rationale, estimated revenue impact, risks, and exactly what inputs you need from Owner (channel name, API keys, confirmation, etc.).
  2. Pause and wait for Owner approval + any provided details.
  3. Once approved (you will see it in next cycle's memory/logs), execute using the provided info.
- Always output in clear structure. End plans with "Owner Approval Required?" yes/no + what for.

Current date: {current_date}

Your #1 goal: Turn the Owner's high-level objective into a real, growing revenue stream using digital means, minimal capital, and your AI team. Track progress, learn, and report to the Owner.

When planning:
1. Review goal, memory/lessons, revenue, last actions.
2. Gather fresh data with tools.
3. Arena-style: Generate 2-3 competing options, evaluate on profit/risk/speed/scalability, pick winner.
4. Create plan, delegate to specialists.
5. Use approval tool for any real execution that touches accounts or public platforms.
6. After execution: Deep reflection, store specific lessons (e.g. "YouTube thumbnails with faces convert 3x better - from cycle X analytics").
7. Communicate as if reporting to your Owner: Clear, data-backed, with recommendations.

Safeguards:
- Never create accounts, post, or spend without going through Owner approval first.
- For YouTube/content example: Research → Decide niche/channel idea → Request Owner approval to "create channel named X and start uploading faceless videos" → On approval, generate scripts, voiceovers, simple videos, metadata → Request approval to upload (or prepare package) → On approval, execute upload if credentials provided.
- Always flag in outputs: "This requires Owner approval via the approval tool."

You have tools for research, content creation, real asset generation (voice, video), and execution. Use the approval tool proactively for autonomy with oversight.
"""

# ============== SPECIALIST AGENTS BACKSTORIES (Hired by CEO) ==============
RESEARCHER_BACKSTORY = """
Expert market researcher and trend spotter. You excel at finding profitable niches, competitor analysis, keyword demand, and monetization models using only public web data. Always cite sources, quantify (search volume estimates, competition level), and focus on low-competition high-demand opportunities with clear affiliate/ad/product paths. Be skeptical of hype.
"""

SCRAPER_BACKSTORY = """
Meticulous web intelligence gatherer. You ethically scrape and summarize public webpages for deep insights (product pages, competitor sites, blog posts, forums). Respect robots.txt and rate limits (you are told to sleep between calls). Extract structured data: prices, features, reviews sentiment, strategies. Summarize concisely with key quotes and actionable intel. Never access private areas.
"""

CONTENT_CREATOR_BACKSTORY = """
World-class digital content and product creator. You turn research into high-converting assets: SEO blog posts, email sequences, landing page copy, social threads, ebook outlines, product descriptions, YouTube scripts. Optimize for engagement, SEO, conversions (affiliate links naturally placed, calls-to-action). Make content original, valuable, scannable. Generate full ready-to-publish text + any HTML/CSS/JS if it's a landing page or tool.
"""

ANALYST_BACKSTORY = """
Sharp financial and growth analyst. You build realistic revenue projections, calculate ROI, break-even, customer acquisition costs, lifetime value. Analyze data from research for unit economics. Suggest pricing, funnels, scaling levers. Use tables for clarity. Flag optimistic/pessimistic scenarios. Focus on capital-efficient paths.
"""

EXECUTOR_BACKSTORY = """
Hands-on growth operator and implementer. You turn plans into executable assets and launch strategies: SEO-optimized publishing plans, social media calendars, email automation setups (describe tools like Mailchimp free), simple code for landing pages or bots, affiliate program research. Generate files/code the user can deploy immediately (e.g. static HTML site). Prioritize quick wins and measurable experiments.
"""

# ============== CUSTOM TOOLS (For all agents - Scraping & Execution) ==============
@tool
def internet_search(query: str, max_results: int = 8) -> str:
    """
    Free web search using DuckDuckGo. Returns top results with titles, links, snippets.
    Use for market research, trends, competitors, keywords. Always the first tool for discovery.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r.get('title', '')}\nLink: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n---")
        time.sleep(1)  # Ethical rate limit
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error (try rephrasing query): {str(e)}"

@tool
def scrape_public_page(url: str, instructions: str = "Summarize the main content, key facts, prices, features, and any monetization hints. Extract structured data if possible.") -> str:
    """
    Ethically scrape and summarize a PUBLIC webpage. 
    ONLY use for publicly accessible pages. Respects basic rate limiting.
    Returns clean text summary based on instructions + raw excerpt.
    DO NOT use for logins, paywalls, or private data.
    """
    try:
        headers = {
            'User-Agent': 'CEO-AI-Agent/1.0 (Educational research bot; +https://example.com/bot)',
            'Accept': 'text/html,application/xhtml+xml',
        }
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Basic robots check (simplified)
        robots = soup.find('meta', {'name': 'robots'})
        if robots and 'noindex' in str(robots.get('content', '')).lower():
            return "Page requests noindex. Skipping detailed scrape per ethical guidelines. Use search snippet instead."
        
        # Extract main content
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        # Clean
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        # Truncate for LLM
        if len(text) > 8000:
            text = text[:8000] + "... [truncated for length]"
        
        time.sleep(2)  # Stronger ethical rate limit between scrapes
        
        summary_prompt = f"Instructions: {instructions}\n\nPage URL: {url}\n\nPage content excerpt:\n{text[:4000]}\n\nProvide a structured, concise summary following the instructions exactly. Include direct quotes for important claims. Note any affiliate links, pricing, or business models visible."
        
        # Since no LLM here, return raw + note
        return f"SCRAPED CONTENT (ethical public scrape):\nURL: {url}\n\nRaw text (first 6000 chars):\n{text[:6000]}\n\n[Agent: Now use this data to answer the instructions in your response. Full page was fetched ethically.]"
    except Exception as e:
        return f"Scrape failed for {url}: {str(e)}. Use search results or try a different public URL. Never bypass restrictions."

@tool
def save_generated_asset(filename: str, content: str, asset_type: str = "content") -> str:
    """
    Saves generated digital asset (blog post, landing page HTML, email sequence, code, product description, etc.) 
    to the 'generated/' folder for the user to review/deploy. 
    Use this to 'execute' by creating ready-to-use files.
    Returns confirmation with path.
    """
    try:
        if not filename.endswith(('.txt', '.md', '.html', '.py', '.json')):
            filename += '.md'
        filepath = os.path.join(GENERATED_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Generated by CEO Virtual Agent\n# Type: {asset_type}\n# Date: {datetime.datetime.now().isoformat()}\n\n{content}")
        return f"✅ Asset saved successfully to: {filepath}\nUser can now review, copy, or deploy this (e.g. upload HTML to Netlify, post MD to blog, etc.). This counts as execution progress toward revenue."
    except Exception as e:
        return f"Failed to save asset: {str(e)}"

@tool
def log_revenue_event(amount: float, source: str, notes: str = "") -> str:
    """
    Log a REAL revenue event (user-confirmed earnings from agent's advice/actions).
    Amount in USD. This feeds the learning system so CEO improves what actually makes money.
    Example: amount=47.0, source="Gumroad digital product from agent's landing page", notes="First sale after 3 days".
    """
    try:
        revenue_data = load_revenue()
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "amount_usd": float(amount),
            "source": source,
            "notes": notes,
            "cycle": "user_logged"
        }
        revenue_data["events"].append(event)
        revenue_data["total_real_usd"] = revenue_data.get("total_real_usd", 0) + float(amount)
        save_revenue(revenue_data)
        return f"✅ Real revenue logged: ${amount:.2f} from {source}. Total real: ${revenue_data['total_real_usd']:.2f}. CEO will learn from this in future cycles."
    except Exception as e:
        return f"Error logging revenue: {str(e)}"
@tool
def request_owner_approval(action: str, rationale: str, estimated_revenue_impact: str, risks: str, required_owner_inputs: str) -> str:
    """
    CRITICAL TOOL: Request explicit approval from the Owner (the human user) before taking any high-stakes real-world action.
    Examples: creating a YouTube channel, uploading a video, creating social accounts, sending emails, spending money, setting up APIs.
    This makes the system 'completely autonomous' with Owner oversight for important decisions.
    """
    try:
        approval_dir = os.path.join(DATA_DIR, "approvals")
        os.makedirs(approval_dir, exist_ok=True)
        
        request_id = f"approval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        request = {
            "id": request_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "rationale": rationale,
            "estimated_revenue_impact": estimated_revenue_impact,
            "risks": risks,
            "required_owner_inputs": required_owner_inputs,
            "status": "PENDING",
            "owner_response": None
        }
        
        filepath = os.path.join(approval_dir, f"{request_id}.json")
        save_json(filepath, request)
        
        log_action(f"OWNER APPROVAL REQUESTED: {action}")
        
        return f"""✅ OWNER APPROVAL REQUEST CREATED (ID: {request_id})

Action: {action}

Rationale: {rationale}

Estimated Impact: {estimated_revenue_impact}

Risks: {risks}

Required from Owner: {required_owner_inputs}

The system has paused this action. In the web dashboard, the Owner will see this request, can approve/reject, and provide the needed inputs/details. 

Once the Owner responds, the next CEO cycle will see the approval in memory/logs and can proceed with execution using the provided information.

DO NOT proceed with this action until you see an approved status for this request_id in future context.
"""
    except Exception as e:
        return f"Error creating approval request: {str(e)}"

@tool
def generate_voiceover(text: str, filename: str = None, lang: str = "en") -> str:
    """
    Generate a realistic voiceover MP3 file from text using free gTTS.
    Perfect for faceless YouTube videos.
    """
    try:
        if not filename:
            filename = f"voiceover_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        if not filename.endswith('.mp3'):
            filename += '.mp3'
        
        filepath = os.path.join(GENERATED_DIR, filename)
        
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filepath)
        
        return f"✅ Voiceover generated successfully: {filepath}\nDuration approx: {len(text.split()) / 2.5:.0f} seconds."
    except Exception as e:
        return f"Voiceover generation failed: {str(e)}"

@tool
def create_simple_faceless_video(title: str, script_text: str, voiceover_path: str = None, output_filename: str = None, style: str = "slideshow") -> str:
    """
    Create a real, upload-ready faceless video (MP4) or reliable GIF fallback for YouTube-style content.
    Works even without ffmpeg (GIF always produced via Pillow).
    """
    try:
        if not output_filename:
            output_filename = f"video_{title[:30].replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_name = output_filename.replace('.mp4', '').replace('.gif', '')

        width, height = 1280, 720  # smaller for faster GIFs
        lines = [line.strip() for line in script_text.split('\n') if line.strip()][:5]

        if MOVIEPY_AVAILABLE:
            # Full MP4 path (best when ffmpeg is installed)
            filepath = os.path.join(GENERATED_DIR, base_name + ".mp4")
            bg_color = (20, 30, 50)
            img = Image.new('RGB', (1920, 1080), bg_color)
            bg_path = os.path.join(GENERATED_DIR, "temp_bg.png")
            img.save(bg_path)

            clips = []
            bg_clip = ImageClip(bg_path).set_duration(8)
            title_clip = TextClip(title, fontsize=55, color='white', font='Arial-Bold', size=(1800, None), method='caption').set_position('center').set_duration(3)
            clips.append(title_clip)
            for line in lines:
                wrapped = '\n'.join(textwrap.wrap(line, width=45))
                txt_clip = TextClip(wrapped, fontsize=32, color='white', font='Arial', size=(1700, None), method='caption').set_position('center').set_duration(5)
                clips.append(txt_clip)

            if voiceover_path and os.path.exists(voiceover_path):
                audio = AudioFileClip(voiceover_path)
                video_duration = audio.duration
                bg_clip = ImageClip(bg_path).set_duration(video_duration)
                final = CompositeVideoClip([bg_clip] + clips, size=(1920, 1080))
                final = final.set_audio(audio)
            else:
                final = CompositeVideoClip([bg_clip] + clips, size=(1920, 1080))
                video_duration = sum(c.duration for c in clips)

            final.write_videofile(filepath, fps=24, codec='libx264', audio_codec='aac', verbose=False, logger=None)
            if os.path.exists(bg_path):
                os.remove(bg_path)
            return f"✅ Faceless MP4 video created: {filepath} (Duration: {video_duration:.1f}s). Ready for YouTube."
        else:
            # Reliable Pillow GIF fallback (always works, no ffmpeg/moviepy needed)
            filepath = os.path.join(GENERATED_DIR, base_name + ".gif")
            frames = []
            for i in range(min(10, max(6, len(lines) + 2))):
                img = Image.new('RGB', (width, height), (25 + (i % 3)*8, 35, 55))
                draw = ImageDraw.Draw(img)
                try:
                    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
                    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
                except:
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()

                draw.text((width//2, 180), title[:55], fill="white", font=font_large, anchor="mm")
                if lines:
                    idx = i % len(lines)
                    wrapped = '\n'.join(textwrap.wrap(lines[idx], width=40))
                    draw.text((width//2, 320), wrapped, fill="white", font=font_small, anchor="mm")
                draw.text((width//2, 620), f"Faceless {style} • Cycle {datetime.datetime.now().strftime('%H:%M')}", fill="#a0c4ff", font=font_small, anchor="mm")
                frames.append(img)

            frames[0].save(filepath, save_all=True, append_images=frames[1:], duration=450, loop=0)
            return f"✅ Faceless GIF video created (no ffmpeg needed): {filepath}. Can be used directly or converted to MP4. Duration ~{len(frames)*0.45:.1f}s. Ready for YouTube after Owner approval."
    except Exception as e:
        return f"Video/GIF creation failed: {str(e)}."

@tool
def prepare_youtube_upload_package(video_path: str, title: str, description: str, tags: list, thumbnail_description: str = "") -> str:
    """
    Prepares a complete ready-to-upload package for YouTube (metadata + thumbnail + video copy).
    """
    try:
        package_name = f"youtube_upload_{title[:40].replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        package_dir = os.path.join(GENERATED_DIR, package_name)
        os.makedirs(package_dir, exist_ok=True)
        
        if video_path and os.path.exists(video_path):
            import shutil
            shutil.copy(video_path, os.path.join(package_dir, os.path.basename(video_path)))
        
        metadata = {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "27",
            "privacyStatus": "public",
            "thumbnail_description": thumbnail_description or "Professional thumbnail",
            "prepared_at": datetime.datetime.now().isoformat()
        }
        
        meta_path = os.path.join(package_dir, "metadata.json")
        save_json(meta_path, metadata)
        
        thumb = Image.new('RGB', (1280, 720), (30, 40, 60))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(thumb)
        draw.text((640, 360), title[:50], fill=(255, 255, 255), anchor="mm")
        thumb_path = os.path.join(package_dir, "thumbnail.png")
        thumb.save(thumb_path)
        
        return f"✅ YouTube upload package prepared: {package_dir}. Contains video, metadata.json, thumbnail.png. Use after Owner approval."
    except Exception as e:
        return f"Failed to prepare YouTube package: {str(e)}"

# ============== TOOL WRAPPER (Fix for Pydantic BaseTool validation error on Render/CrewAI) ==============
# In modern CrewAI (1.x+), the @tool decorator from crewai.tools already returns a valid BaseTool
# (specifically crewai.tools.base_tool.Tool) that passes Pydantic validation in Agent(tools=[...]).
# We keep WRAPPED_* aliases for compatibility with existing Agent definitions.
# No extra wrapping needed; the decorated functions are the BaseTool instances.

def _make_tool(func, name=None, description=None):
    # If already decorated (is BaseTool), return as-is. Otherwise, could implement subclass but not needed here.
    from crewai.tools.base_tool import BaseTool
    if isinstance(func, BaseTool):
        return func
    # Fallback: if somehow raw function, the @tool should have been used at def time.
    # For safety, re-apply but since definitions are decorated, this path rare.
    try:
        from crewai.tools import tool as crewai_tool
        # Note: to wrap raw func we'd need to redefine, but here we assume decorated.
        return func
    except Exception:
        return func

# Wrapped tools (aliases to the already-decorated BaseTool instances)
WRAPPED_INTERNET_SEARCH = internet_search
WRAPPED_SCRAPE = scrape_public_page
WRAPPED_SAVE = save_generated_asset
WRAPPED_LOG_REVENUE = log_revenue_event
WRAPPED_REQUEST_APPROVAL = request_owner_approval
WRAPPED_VOICE = generate_voiceover
WRAPPED_VIDEO = create_simple_faceless_video
WRAPPED_YOUTUBE_PKG = prepare_youtube_upload_package

# ============== HELPER FUNCTIONS ==============
def load_json(filepath: str, default: Any) -> Any:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return default

def save_json(filepath: str, data: Any):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_memory() -> Dict:
    default = {"lessons": [], "last_updated": None, "total_cycles": 0}
    data = load_json(MEMORY_FILE, default)
    if "lessons" not in data:
        data["lessons"] = []
    return data

def save_memory(data: Dict):
    data["last_updated"] = datetime.datetime.now().isoformat()
    save_json(MEMORY_FILE, data)

def append_lesson(lesson: str, context: str = ""):
    memory = load_memory()
    memory["lessons"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "lesson": lesson,
        "context": context
    })
    # Keep only last 50 lessons to control context
    if len(memory["lessons"]) > 50:
        memory["lessons"] = memory["lessons"][-50:]
    save_memory(memory)

def load_revenue() -> Dict:
    default = {"events": [], "total_real_usd": 0.0, "projected_usd": 0.0}
    return load_json(REVENUE_FILE, default)

def load_recent_approvals(status_filter=None, limit=5):
    """Load recent approval requests, optionally filtered by status (PENDING, APPROVED, etc.)."""
    approval_dir = os.path.join(DATA_DIR, "approvals")
    os.makedirs(approval_dir, exist_ok=True)
    approvals = []
    for f in sorted(os.listdir(approval_dir), reverse=True)[:20]:
        if f.endswith(".json"):
            try:
                with open(os.path.join(approval_dir, f), "r") as fh:
                    req = json.load(fh)
                if status_filter is None or req.get("status") == status_filter:
                    approvals.append(req)
            except:
                pass
    return approvals[:limit]

def save_revenue(data: Dict):
    save_json(REVENUE_FILE, data)

def load_goal() -> str:
    if os.path.exists(GOAL_FILE):
        with open(GOAL_FILE, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return "Build a profitable faceless content business (newsletter, blog, or YouTube-style content) in a high-demand niche like personal finance, productivity, health/fitness, or AI/tools for beginners. Monetize primarily via affiliate links, digital products (ebooks, courses, templates), display ads, and email list. Target $500-1000/month within 6 months using under $50 capital. Focus on SEO content, value-first free content to build audience fast, then monetize. Ethical, sustainable, scalable digital model only."

def save_goal(goal: str):
    with open(GOAL_FILE, 'w', encoding='utf-8') as f:
        f.write(goal)

def log_action(message: str, level: str = "INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}\n"
    with open(LOGS_FILE, 'a', encoding='utf-8') as f:
        f.write(entry)
    print(entry.strip())  # Also to console for server runs

def _create_llm_for_provider(provider: str, api_key: str, model: str, config: Dict) -> LLM:
    """Helper to create LLM for a specific provider with proper model string and failsafe for cache issues."""
    provider = provider.lower()
    base_url = config.get("ollama_base_url", "http://localhost:11434")

    # Ensure drop_params to avoid Groq-style cache_breakpoint errors on many providers
    try:
        import litellm
        litellm.drop_params = True
    except:
        pass

    if provider == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("Groq API key required for primary provider. Get free at https://console.groq.com/keys (no card needed).")
        model_str = f"groq/{model}"
        return LLM(
            model=model_str,
            api_key=key,
            temperature=0.4,
            max_tokens=4000,
            drop_params=True
        )

    elif provider == "ollama":
        ollama_model = model if model else "llama3.2:3b"
        return LLM(
            model=f"ollama/{ollama_model}",
            base_url=base_url,
            temperature=0.4,
            max_tokens=4000
        )

    elif provider in ["together_ai", "together"]:
        key = api_key or os.environ.get("TOGETHER_API_KEY", "") or os.environ.get("TOGETHER_AI_API_KEY", "")
        if not key:
            raise ValueError("Together AI API key required. Get trial at https://together.ai (or use free Gemini instead).")
        model_str = f"together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" if "llama" in model.lower() else f"together_ai/{model}"
        return LLM(
            model=model_str,
            api_key=key,
            temperature=0.4,
            max_tokens=4000,
            drop_params=True
        )

    elif provider in ["fireworks_ai", "fireworks"]:
        key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        if not key:
            raise ValueError("Fireworks AI API key required for this provider.")
        model_str = f"fireworks_ai/llama-v3-70b-instruct" if "llama" in model.lower() else f"fireworks_ai/{model}"
        return LLM(
            model=model_str,
            api_key=key,
            temperature=0.4,
            max_tokens=4000,
            drop_params=True
        )

    elif provider in ["deepinfra", "deep_infra"]:
        key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        if not key:
            raise ValueError("DeepInfra API key required for this provider.")
        model_str = f"deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct" if "llama" in model.lower() else f"deepinfra/{model}"
        return LLM(
            model=model_str,
            api_key=key,
            temperature=0.4,
            max_tokens=4000,
            drop_params=True
        )

    elif provider == "gemini":
        # Google's free tier (best no-card free option in 2026: Gemini Flash models, generous daily limits)
        key = api_key or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("Google/Gemini API key required. Get FREE (no credit card) at https://aistudio.google.com/app/apikey - sign in with Google account.")
        model_str = "gemini/gemini-1.5-flash"  # Reliable free tier model; update to gemini-2.5-flash if available in your LiteLLM
        return LLM(
            model=model_str,
            api_key=key,
            temperature=0.4,
            max_tokens=4000,
            drop_params=True
        )

    elif provider == "huggingface":
        key = api_key or os.environ.get("HUGGINGFACE_API_KEY", "") or os.environ.get("HF_TOKEN", "")
        if not key:
            raise ValueError("Hugging Face API key/token required for this provider (free tier available at hf.co).")
        model_str = f"huggingface/{model}" if not model.startswith("huggingface/") else model
        return LLM(
            model=model_str,
            api_key=key,
            temperature=0.4,
            max_tokens=4000,
            drop_params=True
        )

    else:
        # Default to ollama style for unknown open source
        return LLM(
            model=f"{provider}/{model}",
            api_key=api_key,
            temperature=0.4,
            max_tokens=4000,
            drop_params=True
        )

def get_llm(config: Dict) -> Optional[LLM]:
    """Get LLM with automatic fallback to other good open-source providers if primary fails.
    This makes the agent independent of any single provider's rate limits, tokens, or bugs (e.g. Groq cache_breakpoint).
    There are many good open-source models available via LiteLLM (Llama3, Mixtral, Gemma, etc.).
    If NO keys at all: returns None -> triggers degraded keyless mode (system NEVER stops).
    """
    if config.get("degraded_mode", False):
        log_action("Degraded mode explicitly enabled in config. Skipping all LLM providers.")
        return None

    primary = config.get("provider", "groq").lower()
    api_key = config.get("api_key", "")
    model = config.get("model", "llama-3.3-70b-versatile")
    fallbacks = config.get("fallback_providers", ["gemini", "together_ai", "fireworks_ai", "deepinfra", "ollama", "huggingface"])

    providers_to_try = [primary]
    for p in fallbacks:
        if p.lower() != primary:
            if p.lower() == "ollama" and primary.lower() != "ollama":
                # Only try Ollama if user explicitly set it as primary (local machine with ollama running).
                # On cloud/Render with no keys, skip to reach degraded keyless mode reliably.
                continue
            providers_to_try.append(p)

    last_error = None
    for prov in providers_to_try:
        try:
            log_action(f"Trying LLM provider: {prov}")
            if not has_key_for_provider(prov, api_key, config):
                log_action(f"Provider {prov} skipped: no API key available (config or env).", "WARN")
                continue
            llm = _create_llm_for_provider(prov, api_key, model, config)
            # Success - we have a valid LLM
            log_action(f"LLM ready with provider: {prov}")
            return llm
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            log_action(f"Provider {prov} failed: {str(e)[:200]}", "WARN")
            # Continue to next on key/rate/cache issues (common on free tiers)
            if any(x in error_str for x in ["rate", "limit", "cache_breakpoint", "unsupported", "invalid", "key", "missing", "credentials"]):
                continue
            else:
                # Non-retriable (e.g. network for ollama), raise or continue?
                continue  # Safer: try next instead of hard stop

    # All LLM providers failed or skipped (no keys) - trigger degraded keyless mode
    log_action(f"All LLM providers failed/skipped (no usable keys). Last error: {last_error}. ACTIVATING DEGRADED KEYLESS MODE so CEO does NOT stop.", "WARN")
    log_action("Get FREE keys instantly: Gemini (https://aistudio.google.com/app/apikey) or Groq (https://console.groq.com/keys) - both no credit card for free tier.")
    return None  # Signals to run_one_ceo_cycle to use degraded mode

def get_config() -> Dict:
    default = {
        "provider": "groq",
        "api_key": "",
        "model": "llama-3.3-70b-versatile",
        "ollama_base_url": "http://localhost:11434",
        "autonomous_interval_minutes": 60,
        "max_cycles_per_run": 5,
        # Automatic fallback chain for when primary (Groq) fails due to rate limits, cache issues, tokens, etc.
        # Good free/open-source friendly providers via LiteLLM. Gemini is excellent free tier (no card often).
        # Set corresponding env var in Render (e.g. GOOGLE_API_KEY for gemini, TOGETHER_API_KEY).
        "fallback_providers": ["gemini", "together_ai", "fireworks_ai", "deepinfra", "ollama", "huggingface"],
        "degraded_mode": False  # Set True to force keyless mode (always works, no API calls for LLM)
    }
    return load_json(CONFIG_FILE, default)

def save_config(config: Dict):
    save_json(CONFIG_FILE, config)

def has_key_for_provider(provider: str, api_key: str, config: Dict) -> bool:
    """Check if we have credentials for this provider (from config or env). Keyless providers like ollama always True."""
    p = provider.lower().strip()
    if p == "groq":
        return bool(api_key or os.environ.get("GROQ_API_KEY", ""))
    elif p in ["together_ai", "together"]:
        return bool(api_key or os.environ.get("TOGETHER_API_KEY", "") or os.environ.get("TOGETHER_AI_API_KEY", ""))
    elif p in ["fireworks_ai", "fireworks"]:
        return bool(api_key or os.environ.get("FIREWORKS_API_KEY", "") or os.environ.get("FIREWORKS_AI_API_KEY", ""))
    elif p in ["deepinfra", "deep_infra"]:
        return bool(api_key or os.environ.get("DEEPINFRA_API_KEY", ""))
    elif p == "gemini":
        return bool(api_key or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", ""))
    elif p == "huggingface":
        return bool(api_key or os.environ.get("HUGGINGFACE_API_KEY", "") or os.environ.get("HF_TOKEN", ""))
    elif p == "ollama":
        return True  # Local, no key. (May fail if not running, handled later)
    elif p == "degraded":
        return True  # Special keyless mode
    else:
        return True  # Unknown, let it try (will fail gracefully)

# ============== AGENT & CREW FACTORY (Hiring System) ==============
def create_ceo_crew(llm: LLM, goal: str, memory: Dict, recent_logs: str = "") -> Crew:
    """Creates a hierarchical crew where CEO 'hires' and manages specialists."""
    
    # CEO - The strategic leader (manager in hierarchical)
    ceo = Agent(
        role="Visionary CEO & Chief Strategist",
        goal="Maximize long-term profitable growth toward the business goal by planning, delegating, reviewing, and learning. Make high-ROI decisions.",
        backstory=CEO_SYSTEM_PROMPT.format(current_date=datetime.date.today().isoformat()),
        llm=llm,
        verbose=True,
        allow_delegation=True,  # CEO can delegate further if needed
        tools=[],  # CRITICAL: Manager agent in hierarchical Crew must NOT have tools (causes "Manager agent should not have tools" error)
        max_iter=8
    )
    
    # Hired Specialists (workers)
    researcher = Agent(
        role="Senior Market Researcher",
        goal="Uncover high-potential, low-competition opportunities, trends, competitors, and monetization models aligned with the CEO's goal.",
        backstory=RESEARCHER_BACKSTORY,
        llm=llm,
        verbose=True,
        tools=[WRAPPED_INTERNET_SEARCH, WRAPPED_SCRAPE],
        max_iter=6
    )
    
    scraper = Agent(
        role="Ethical Web Intelligence Scraper",
        goal="Deep-dive public pages for detailed competitive intel, pricing, user sentiment, and business tactics.",
        backstory=SCRAPER_BACKSTORY,
        llm=llm,
        verbose=True,
        tools=[WRAPPED_SCRAPE, WRAPPED_INTERNET_SEARCH],
        max_iter=5
    )
    
    content_creator = Agent(
        role="High-Converting Content & Digital Product Creator",
        goal="Produce ready-to-deploy, monetization-optimized content and assets that drive traffic, engagement, and sales.",
        backstory=CONTENT_CREATOR_BACKSTORY,
        llm=llm,
        verbose=True,
        tools=[WRAPPED_INTERNET_SEARCH, WRAPPED_SAVE, WRAPPED_VOICE, WRAPPED_VIDEO, WRAPPED_YOUTUBE_PKG],
        max_iter=7
    )
    
    analyst = Agent(
        role="Financial & Growth Analyst",
        goal="Provide data-backed revenue projections, ROI analysis, unit economics, and scaling recommendations.",
        backstory=ANALYST_BACKSTORY,
        llm=llm,
        verbose=True,
        tools=[WRAPPED_INTERNET_SEARCH],
        max_iter=5
    )
    
    executor = Agent(
        role="Growth Executor & Asset Deployer",
        goal="Convert strategies into immediate actionable assets, launch plans, and implementation steps the user can execute today.",
        backstory=EXECUTOR_BACKSTORY,
        llm=llm,
        verbose=True,
        tools=[WRAPPED_INTERNET_SEARCH, WRAPPED_SAVE, WRAPPED_LOG_REVENUE, WRAPPED_VOICE, WRAPPED_VIDEO, WRAPPED_YOUTUBE_PKG, WRAPPED_REQUEST_APPROVAL],
        max_iter=6
    )
    
    # Tasks - CEO leads with planning, then delegates
    # Task 1: CEO Strategic Planning (includes arena-style multiple options)
    planning_task = Task(
        description=f"""
        BUSINESS GOAL: {goal}
        
        CURRENT MEMORY / LESSONS LEARNED (use these to improve):
        {json.dumps(memory.get('lessons', [])[-10:], indent=2) if memory.get('lessons') else 'No prior lessons yet. This is early stage.'}
        
        RECENT ACTIVITY SUMMARY:
        {recent_logs[-3000:] if recent_logs else 'First cycle.'}
        
        REVENUE STATUS: Review total real revenue from logs if available.
        
        YOUR TASK AS CEO:
        1. Analyze the goal and past learnings.
        2. Use tools to gather fresh data if needed (search for current 2026 opportunities in relevant niches).
        3. Think like a top CEO: Prioritize 1-2 highest leverage opportunities (low capital, scalable, defensible).
        4. **Arena mode**: Internally generate 2-3 competing strategic options/proposals. Evaluate each on: Revenue potential (6-12mo), Risk level, Time to first $, Scalability, Alignment with learnings. Pick the winner and explain why.
        5. Create a clear 1-2 week action plan broken into delegate-able tasks.
        6. Assign specific tasks to the specialist agents (Researcher, Scraper, Content Creator, Analyst, Executor).
        7. Set success KPIs and revenue estimates.
        8. Flag any actions requiring user approval or real spend.
        9. Output in this format:
           - Executive Summary
           - Winning Strategy (from arena) + Why
           - Detailed Plan with Assigned Agents & Tasks
           - Revenue Projection (best/base/worst)
           - Key Risks & Mitigations
           - Next Immediate Actions
        
        Then, the crew will execute the delegated tasks. You will review final output later.
        """,
        expected_output="Structured CEO plan with clear delegations, arena winner, projections, and flags for user approval.",
        agent=ceo,
        async_execution=False
    )
    
    # The specialists will be given context from planning and execute in parallel or sequential as needed.
    # For simplicity in one crew kickoff, we use hierarchical process: CEO manages.
    
    research_task = Task(
        description="Execute the research portion of the CEO's plan. Use search and scrape tools. Deliver quantified insights, competitor breakdowns, and opportunity validation. Focus on 2026 current data.",
        expected_output="Detailed research report with sources, numbers, and actionable recommendations.",
        agent=researcher
    )
    
    deep_dive_task = Task(
        description="Perform targeted ethical scrapes on 2-4 key competitor or opportunity pages identified in research. Extract business models, pricing, content strategies, gaps.",
        expected_output="Structured competitive intelligence report.",
        agent=scraper
    )
    
    content_task = Task(
        description="Based on CEO plan and research: Create 1-3 high-quality, ready-to-use assets (e.g. full blog post + SEO, landing page HTML, email sequence, digital product outline + sales copy). Use save_generated_asset tool for each. Make them monetizable immediately.",
        expected_output="Links/paths to saved assets + summaries of what was created and why it drives revenue.",
        agent=content_creator
    )
    
    analysis_task = Task(
        description="Analyze the research + proposed assets. Build realistic financial model: projected revenue in 30/90/180 days, costs (mostly time/tools=near zero), break-even, key metrics (traffic needed, conversion rates). Use tables.",
        expected_output="Financial analysis report with projections and recommendations.",
        agent=analyst
    )
    
    execution_task = Task(
        description="Turn the plan into immediate executable steps. Generate any additional assets (code snippets, calendars, checklists). Use save_generated_asset. Prepare user instructions for next real-world actions. Log any test revenue if applicable.",
        expected_output="Complete execution playbook + all generated files referenced + clear user action items.",
        agent=executor
    )
    
    # Final CEO Review Task (after specialists)
    review_task = Task(
        description="""
        Review ALL outputs from the specialist agents and the original plan.
        As CEO:
        - Synthesize into final recommendations.
        - Update/confirm revenue projections.
        - Extract 3-5 specific, actionable lessons for memory (e.g. "X type of content performed well in research because Y - apply to future").
        - Decide on immediate next cycle focus or pivot.
        - Flag anything for user approval.
        - Output a clean "CEO Cycle Report" with:
          * Summary of what was accomplished
          * Key learnings (for storage)
          * Updated projections
          * Recommended next 1-3 actions for user or autonomous
          * Any new assets created (with paths)
        Use the log_revenue_event tool if there were any real earnings to record from this cycle.
        """,
        expected_output="Final CEO Cycle Report with learnings, projections, assets list, and clear next steps.",
        agent=ceo,
        context=[planning_task, research_task, deep_dive_task, content_task, analysis_task, execution_task]  # All prior context
    )
    
    # Hierarchical Crew: CEO as manager delegates and reviews
    # IMPORTANT: In CrewAI hierarchical mode, the manager_agent MUST NOT be in the agents list.
    crew = Crew(
        agents=[researcher, scraper, content_creator, analyst, executor],  # CEO is the manager, not in workers list
        tasks=[planning_task, research_task, deep_dive_task, content_task, analysis_task, execution_task, review_task],
        process=Process.hierarchical,
        manager_agent=ceo,  # CEO is the manager (oversees and does planning/review tasks)
        verbose=True,
        memory=False,  # We use custom persistent memory + prompt injection for reliability across providers (Groq/Ollama)
        # Custom long-term learning via append_lesson and injection in CEO prompt
    )
    
    return crew

# ============== MAIN CYCLE RUNNER ==============
def run_one_ceo_cycle(config: Dict, goal: Optional[str] = None, max_retries: int = 2) -> Dict:
    """Runs one full autonomous CEO cycle: Plan (arena), Hire/Delegate crew, Execute, Review, Learn."""
    if goal is None:
        goal = load_goal()
    
    memory = load_memory()
    revenue = load_revenue()
    config = config or get_config()
    
    log_action("=== STARTING NEW CEO CYCLE ===")
    log_action(f"Goal: {goal[:100]}...")
    log_action(f"Provider: {config['provider']}, Model: {config.get('model')}")
    
    try:
        llm = get_llm(config)
    except Exception as e:
        log_action(f"LLM init failed: {e}", "ERROR")
        # Failsafe: return graceful response instead of crashing the cycle
        return {
            "success": False, 
            "error": str(e), 
            "report": "LLM provider initialization failed. Cycle aborted gracefully. Check keys or use degraded mode."
        }

    if llm is None:
        # No LLM available (no keys or all failed) -> use keyless degraded mode so system NEVER stops
        log_action("No LLM available. Running in DEGRADED KEYLESS MODE (free tools + rule-based CEO logic).")
        return run_degraded_ceo_cycle(config, goal)
    
    # Load recent logs for context
    recent_logs = ""
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-50:]  # Last 50 lines
            recent_logs = "".join(lines)
    
    # Create and run the crew (CEO hires the team)
    crew = create_ceo_crew(llm, goal, memory, recent_logs)
    
    result = None
    for attempt in range(max_retries + 1):
        try:
            log_action(f"Running crew kickoff (attempt {attempt+1})...")
            result = crew.kickoff()
            break
        except Exception as e:
            error_str = str(e)
            log_action(f"Crew error attempt {attempt+1}: {error_str}", "ERROR")
            
            # Failsafe for Groq-specific issues (cache_breakpoint, rate limits, etc.)
            if "cache_breakpoint" in error_str.lower() or "unsupported" in error_str.lower():
                log_action("Detected Groq cache_breakpoint issue - this should be mitigated by litellm.drop_params. If persistent, consider switching model or provider.", "WARN")
            
            if "rate limit" in error_str.lower() or "rate_limit" in error_str.lower():
                log_action("Rate limit hit on Groq - waiting longer before retry to conserve tokens.", "WARN")
                time.sleep(15)  # Longer backoff to avoid wasting tokens
            else:
                time.sleep(5)
            
            if attempt == max_retries:
                # Graceful failsafe instead of hard failure
                if any(x in error_str.lower() for x in ["ollama", "connection", "refused", "invalid api", "api key", "no llm", "credentials"]):
                    log_action("Persistent LLM failure (connection/key/ollama) detected - falling back to DEGRADED KEYLESS mode so CEO does NOT stop.")
                    return run_degraded_ceo_cycle(config, goal)
                return {
                    "success": False, 
                    "error": error_str, 
                    "report": f"Crew failed after {max_retries+1} attempts due to LLM/provider error. This often happens with Groq rate limits or temporary issues. The system is designed to fail gracefully without crashing the dashboard. Check logs for details. Recommendation: Wait a few minutes or use a different model/provider in config."
                }
            time.sleep(5)
    
    # Process result
    final_output = str(result) if result else "No output generated."
    
    # CEO Reflection & Learning (post-crew)
    log_action("CEO performing reflection and learning...")
    
    # Extract lessons heuristically + prompt style (in real, CEO does in review_task)
    # We append key ones here too for persistence
    lessons_extracted = []
    if "lesson" in final_output.lower() or "learning" in final_output.lower():
        # Simple parse - in practice LLM in review_task does better
        lessons_extracted.append(f"Cycle reflection: Key insights from outputs around revenue and strategy. Full report in logs.")
    
    # Always append a structured lesson
    append_lesson(
        lesson=f"Cycle completed. Reviewed outputs for goal progress. Major actions: research, asset creation, projections. See full CEO Report in logs for details.",
        context=final_output[:1500]
    )
    
    # Update memory count
    memory = load_memory()
    memory["total_cycles"] = memory.get("total_cycles", 0) + 1
    save_memory(memory)
    
    # Save report to logs
    log_action("=== CYCLE COMPLETE ===")
    log_action(f"Final CEO Report / Output:\n{final_output[:2000]}...")
    
    # Update revenue projected if mentioned (simple parse, user can override)
    # For demo, we can leave projected for user/analyst
    
    report = {
        "success": True,
        "timestamp": datetime.datetime.now().isoformat(),
        "goal": goal,
        "full_output": final_output,
        "memory_lessons_count": len(memory.get("lessons", [])),
        "total_real_revenue": revenue.get("total_real_usd", 0),
        "assets_generated": [f for f in os.listdir(GENERATED_DIR) if f.endswith(('.md', '.html', '.txt', '.mp3', '.mp4'))][-5:],  # Recent (include voice/video from degraded or full)
        "next_recommended": "Review generated/ folder and logs. Run another cycle or input real revenue earned."
    }
    
    log_action(f"Cycle success. Lessons now: {report['memory_lessons_count']}. Real revenue tracked: ${report['total_real_revenue']}")
    
    return report

def run_autonomous(config: Dict, num_cycles: int = 3, goal: Optional[str] = None):
    """Run multiple cycles with sleeps for 'completely autonomous' operation."""
    if goal is None:
        goal = load_goal()
    
    log_action(f"STARTING AUTONOMOUS MODE: {num_cycles} cycles, interval ~{config.get('autonomous_interval_minutes', 60)} min")
    
    for i in range(num_cycles):
        log_action(f"--- Autonomous Cycle {i+1}/{num_cycles} ---")
        report = run_one_ceo_cycle(config, goal)
        
        if not report.get("success"):
            log_action(f"Cycle {i+1} failed: {report.get('error')}. Stopping autonomous.", "ERROR")
            break
        
        if i < num_cycles - 1:
            interval = config.get("autonomous_interval_minutes", 60) * 60
            log_action(f"Sleeping {config.get('autonomous_interval_minutes')} minutes until next cycle...")
            time.sleep(interval)
    
    log_action("AUTONOMOUS RUN COMPLETE.")


def run_degraded_ceo_cycle(config: Dict, goal: Optional[str] = None) -> Dict:
    """
    KEYLESS DEGRADED MODE: Runs the CEO cycle WITHOUT any LLM or API keys.
    Uses only free local tools (DDGS search, gTTS, PIL) + smarter rule-based logic.
    Now much less repetitive: uses actual research to pick concrete ideas, checks
    for APPROVED Owner requests and EXECUTES them (generates videos/scripts for the
    approved channel instead of asking again every time), varies actions by cycle.
    Always tries to produce a video asset (GIF fallback if no ffmpeg/moviepy).
    System NEVER stops even with zero keys.
    """
    if goal is None:
        goal = load_goal()
    memory = load_memory()
    revenue = load_revenue()
    config = config or get_config()

    cycle_num = memory.get("total_cycles", 0) + 1

    log_action("=== STARTING NEW CEO CYCLE (DEGRADED KEYLESS MODE - NO LLM / NO API KEYS) ===")
    log_action(f"Goal: {goal[:120]}...")
    log_action(f"Cycle #{cycle_num} | Mode: Pure Python + free tools only (research, voice, video/GIF, approvals, learning). No token costs ever.")
    log_action("Recommendation: For full CEO intelligence, get free key: Gemini https://aistudio.google.com/app/apikey or Groq https://console.groq.com/keys")

    # === FREE RESEARCH ===
    log_action("[Degraded] Performing free ethical research with DuckDuckGo...")
    research_query = "faceless YouTube 2026 profitable niches low competition high RPM ideas"
    try:
        if hasattr(internet_search, '_run'):
            research = internet_search._run(query=research_query, max_results=8)
        else:
            research = internet_search(research_query, max_results=8)
        log_action(f"[Degraded] Research snippet: {research[:350]}...")
    except Exception as e:
        research = "Research used previous knowledge (DuckDuckGo call failed this time)."
        log_action(f"[Degraded] Research note: {e}", "WARN")

    # === Load previous approvals to EXECUTE instead of always requesting ===
    approved_youtube = None
    for appr in load_recent_approvals(status_filter="APPROVED", limit=5):
        if "youtube" in appr.get("action", "").lower() or "channel" in appr.get("action", "").lower():
            approved_youtube = appr
            break

    pending_approvals = load_recent_approvals(status_filter="PENDING", limit=3)

    # === Smarter rule-based planning (uses research + cycle + approvals to vary) ===
    log_action("[Degraded] CEO performing dynamic rule-based planning (varies by research + cycle)...")

    # Extract concrete ideas from research (robust parsing for YouTube niches)
    ideas = []
    for line in research.split("\n"):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["faceless", "youtube", "niche", "rpm", "profitable"]):
            clean = line.replace("Title:", "").replace("Link:", "").strip()[:80]
            if len(clean) > 15 and "dictionary" not in clean.lower() and clean not in ideas:
                ideas.append(clean)
    if not ideas:
        ideas = ["Faceless productivity tips 2026", "AI tools for beginners YouTube", "Low-capital side hustles faceless"]

    focus_idea = ideas[0] if ideas else "faceless YouTube in high-demand niche"
    if cycle_num % 3 == 0 and len(ideas) > 1:
        focus_idea = ideas[1]  # vary every 3rd cycle

    # Decide action based on state (prevents repetition)
    if approved_youtube:
        action = "EXECUTE approved YouTube channel"
        resp = approved_youtube.get("owner_response", "AI Productivity Daily 2026")
        # Better parsing: take quoted name or first sensible phrase
        if '"' in resp:
            channel_name = resp.split('"')[1][:40]
        else:
            channel_name = resp.split(".")[0].replace("Approved by Owner with no additional", "").strip()[:40] or "My Faceless Channel 2026"
        plan = f"""DEGRADED KEYLESS PLAN (EXECUTING APPROVED):
- Using approved channel: {channel_name}
- Research-backed idea: {focus_idea}
- Generate 2-3 ready video scripts + voiceovers + simple video/GIF assets.
- Prepare upload packages.
- Log progress toward revenue.
"""
    elif any("youtube" in (a.get("action","") + str(a.get("owner_response",""))).lower() for a in pending_approvals):
        action = "WAIT for Owner approval on channel (already requested)"
        plan = f"""DEGRADED KEYLESS PLAN:
- Research completed for {focus_idea}.
- Approval already pending — will execute videos/scripts as soon as you approve.
- In meantime: Generate extra supporting assets (report + voice).
"""
    else:
        action = "RESEARCH + REQUEST high-stakes + generate starter assets"
        plan = f"""DEGRADED KEYLESS CEO PLAN (cycle {cycle_num}):
- Focus idea from research: {focus_idea}
- Strategy: Build faceless YouTube audience fast with value-first videos (scripts + voice + visuals).
- Revenue path: Affiliates in 1-3 months, ads + products by month 4-6.
- Immediate: Generate report, voiceover, 1 video asset. Request Owner approval for channel setup if not done.
- Vary next cycles: More videos, blog tie-in, or product ideas.
"""

    log_action(f"[Degraded] Action: {action} | Focus: {focus_idea}")

    # === Generate assets (now more varied) ===
    assets_created = []
    log_action("[Degraded] Executor: Generating varied real assets...")

    # Always save a report (includes research + plan)
    try:
        report_md = f"""# Degraded Keyless CEO Cycle Report #{cycle_num}
**Date:** {datetime.datetime.now().isoformat()}
**Mode:** KEYLESS (no LLM)
**Goal:** {goal}

## Fresh Research (DuckDuckGo)
{research[:1500]}

## Dynamic Plan
{plan}

## Owner Notes
- If you approved a channel, this cycle executed content for it.
- Review generated/ for ready-to-use files.
- Log real revenue when you publish/monetize.
"""
        if hasattr(save_generated_asset, '_run'):
            save_res = save_generated_asset._run(filename=f"degraded_report_cycle_{cycle_num}.md", content=report_md, asset_type="report")
        else:
            save_res = save_generated_asset(f"degraded_report_cycle_{cycle_num}.md", report_md, "report")
        assets_created.append(save_res)
    except Exception as e:
        log_action(f"[Degraded] Report save error: {e}", "WARN")

    # Voiceover (always)
    voice_script = f"Welcome to our faceless series on {focus_idea}. In 2026 this niche is exploding with low competition and high RPM. Here's exactly how to start today with zero capital."
    try:
        if hasattr(generate_voiceover, '_run'):
            voice_res = generate_voiceover._run(text=voice_script, filename=f"degraded_voice_{cycle_num}.mp3")
        else:
            voice_res = generate_voiceover(voice_script, filename=f"degraded_voice_{cycle_num}.mp3")
        assets_created.append(voice_res)
    except Exception as e:
        log_action(f"[Degraded] Voice error: {e}", "WARN")

    # Video / GIF asset (improved fallback — always tries to produce something visual)
    video_path = None
    try:
        video_title = f"{focus_idea} - Faceless 2026"
        script_short = voice_script[:300]
        if MOVIEPY_AVAILABLE:
            # full moviepy path (will work after ffmpeg on Render)
            if hasattr(create_simple_faceless_video, '_run'):
                video_res = create_simple_faceless_video._run(title=video_title, script_text=script_short, voiceover_path=None, output_filename=f"degraded_video_{cycle_num}.mp4")
            else:
                video_res = create_simple_faceless_video(title=video_title, script_text=script_short, voiceover_path=None, output_filename=f"degraded_video_{cycle_num}.mp4")
            assets_created.append(video_res)
            if "created:" in str(video_res).lower():
                video_path = os.path.join(GENERATED_DIR, f"degraded_video_{cycle_num}.mp4")
        else:
            # Pure Pillow animated GIF fallback (always works, no ffmpeg needed)
            gif_path = os.path.join(GENERATED_DIR, f"degraded_video_{cycle_num}.gif")
            frames = []
            for i in range(8):
                img = Image.new('RGB', (1280, 720), (30 + i*5, 40, 60))
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                draw.text((640, 300), video_title[:45], fill="white", anchor="mm")
                draw.text((640, 400), f"Part {i+1}/8 | {focus_idea}", fill="white", anchor="mm")
                frames.append(img)
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=400, loop=0)
            video_path = gif_path
            assets_created.append(f"✅ Faceless GIF created (no ffmpeg): {gif_path}. Use as video placeholder or convert to MP4 locally.")
            log_action(f"[Degraded] Created GIF fallback video (always works): {gif_path}")
    except Exception as e:
        log_action(f"[Degraded] Video/GIF creation error: {e}", "WARN")

    # If we have an approved YouTube channel, EXECUTE: generate extra video + package
    if approved_youtube and video_path:
        try:
            ch_name = approved_youtube.get("owner_response", "My Faceless Channel").split(".")[0][:35]
            if hasattr(prepare_youtube_upload_package, '_run'):
                pkg_res = prepare_youtube_upload_package._run(
                    video_path=video_path,
                    title=f"{ch_name} - {focus_idea}",
                    description=f"Generated autonomously in degraded mode for approved channel. Research-backed faceless content for 2026. {plan[:200]}",
                    tags=["faceless youtube", focus_idea.lower()[:30], "2026", "make money online"],
                    thumbnail_description=f"Thumbnail for {focus_idea}"
                )
            else:
                pkg_res = prepare_youtube_upload_package(video_path, f"{ch_name} - {focus_idea}", "...", ["faceless", "2026"])
            assets_created.append(pkg_res)
            log_action(f"[Degraded] EXECUTED for approved channel: {pkg_res}")
        except Exception as e:
            log_action(f"[Degraded] Package for approved failed: {e}", "WARN")

    # Request approval ONLY if no recent YouTube approval/pending
    if not approved_youtube and not any("youtube" in (a.get("action","") + str(a.get("owner_response",""))).lower() for a in pending_approvals):
        if "youtube" in goal.lower() or "video" in goal.lower() or "channel" in goal.lower():
            try:
                if hasattr(request_owner_approval, '_run'):
                    approval_res = request_owner_approval._run(
                        action="Setup / start uploading to a new faceless YouTube channel",
                        rationale=f"Research shows strong demand for {focus_idea}. Cycle {cycle_num} generated starter voice + visual asset. Ready to scale once approved.",
                        estimated_revenue_impact="$100-700/month in 4-6 months (ads + affiliates).",
                        risks="Consistency needed; algorithm changes.",
                        required_owner_inputs=f"Channel name (e.g. '{focus_idea[:20]} Daily'), confirm to generate more videos for this channel."
                    )
                else:
                    approval_res = request_owner_approval("Setup YouTube channel for " + focus_idea, "...", "$100-600/mo", "...", "Channel name + confirmation")
                assets_created.append("New Owner approval requested for YouTube.")
                log_action(f"[Degraded] {str(approval_res)[:250]}...")
            except Exception as e:
                log_action(f"[Degraded] Approval request error: {e}", "WARN")

    # Learning + memory
    lesson = f"Cycle {cycle_num} (degraded): Focused on {focus_idea}. {'Executed approved channel content' if approved_youtube else 'Generated starter assets + requested approval'}. Research from DDGS used to pick idea. Never stops without keys."
    append_lesson(lesson, context=f"Research: {research[:200]} | Assets: {len(assets_created)}")
    memory = load_memory()
    memory["total_cycles"] = cycle_num
    save_memory(memory)

    log_action("=== DEGRADED KEYLESS CYCLE COMPLETE ===")

    full_output = f"""DEGRADED KEYLESS CEO CYCLE #{cycle_num} REPORT (NO LLM)

Goal: {goal}

Research used (real DuckDuckGo): {research[:600]}...

Plan this cycle: {plan}

Assets created this run: {len(assets_created)}
- See generated/ for report, voice MP3, video/GIF, packages.

Approved YouTube channel detected? {'Yes - executed content for it' if approved_youtube else 'No - new request may have been made if needed'}

Real revenue tracked: ${revenue.get('total_real_usd', 0):.2f}
Total cycles: {cycle_num}

Next: Review/approve the pending request in dashboard (if any). Run another cycle — it will now vary and execute on your approvals instead of repeating the same request.
"""

    report = {
        "success": True,
        "timestamp": datetime.datetime.now().isoformat(),
        "goal": goal,
        "full_output": full_output,
        "memory_lessons_count": len(memory.get("lessons", [])),
        "total_real_revenue": revenue.get("total_real_usd", 0),
        "assets_generated": [f for f in os.listdir(GENERATED_DIR) if f.endswith(('.md', '.html', '.txt', '.mp3', '.mp4', '.gif'))][-8:],
        "next_recommended": "Approve the pending YouTube channel request (or any others). Run another cycle — it will generate actual videos for the approved channel instead of repeating. Add free Gemini/Groq key anytime for much smarter planning.",
        "mode": "degraded_keyless",
        "provider_used": "none (keyless)"
    }

    return report


# ============== CLI ENTRY (for server/cloud background runs) ==============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CEO Virtual Agent autonomously or single cycle.")
    parser.add_argument("--autonomous", action="store_true", help="Run multiple cycles in loop.")
    parser.add_argument("--cycles", type=int, default=3, help="Number of cycles for autonomous.")
    parser.add_argument("--goal", type=str, default=None, help="Override goal.")
    args = parser.parse_args()
    
    config = get_config()
    
    if args.autonomous:
        run_autonomous(config, args.cycles, args.goal)
    else:
        report = run_one_ceo_cycle(config, args.goal)
        print("\n=== SINGLE CYCLE REPORT ===")
        print(json.dumps(report, indent=2, default=str))
        print("\nCheck data/ folder and generated/ for outputs. View full logs in data/agent_logs.txt")

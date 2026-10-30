"""
AI Quiz and Flashcard Generator
--------------------------------
A Streamlit app that turns lesson notes into MCQ quizzes and flashcards
using the Groq LLM API.

Run with:  streamlit run app.py
"""

import html
import json
import os
import re

import streamlit as st
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    Groq,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGES = ["Home", "Generate Quiz", "Flashcards", "Quiz Results"]
PAGE_ICONS = {
    "Home": "🏠",
    "Generate Quiz": "📝",
    "Flashcards": "🃏",
    "Quiz Results": "🏆",
}
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
LETTERS = ["A", "B", "C", "D"]
MAX_NOTES_CHARS = 12000  # keeps the prompt within a safe size

DIFFICULTY_HINTS = {
    "Easy": "Simple recall questions about basic facts and definitions stated directly in the notes.",
    "Medium": "Questions that need understanding of concepts and the relationships between ideas in the notes.",
    "Hard": "Challenging questions that need analysis, comparison or application of the ideas in the notes, "
    "with plausible, closely related distractors.",
}

# PDF upload is optional: it only works if the pypdf package is installed.
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

st.set_page_config(
    page_title="AI Quiz & Flashcard Generator",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; max-width: 1050px;}
        .hero {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #ec4899 100%);
            padding: 2rem 2.2rem; border-radius: 20px; color: white;
            box-shadow: 0 10px 30px rgba(79,70,229,0.25); margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0; font-size: 2.1rem; color: white;}
        .hero p {margin: 0.5rem 0 0 0; font-size: 1.05rem; opacity: 0.95; color: white;}
        .feature-card {
            background: white; border-radius: 16px; padding: 1.2rem 1.3rem;
            border: 1px solid #e5e7eb; box-shadow: 0 4px 14px rgba(0,0,0,0.05);
            height: 100%; color: #1f2937;
        }
        .feature-card h3 {margin: 0 0 0.4rem 0; color: #4f46e5; font-size: 1.15rem;}
        .feature-card p {margin: 0; color: #4b5563; font-size: 0.95rem;}
        .question-card {
            background: #f5f3ff; border-left: 6px solid #7c3aed;
            padding: 1.1rem 1.3rem; border-radius: 12px; margin: 0.6rem 0 1rem 0;
            font-size: 1.15rem; font-weight: 600; color: #1f2937;
        }
        .flashcard {
            border-radius: 22px; padding: 2.5rem 2rem; min-height: 230px;
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; text-align: center; color: white;
            box-shadow: 0 10px 28px rgba(0,0,0,0.18); margin: 1rem 0;
        }
        .flashcard.front {background: linear-gradient(135deg, #4f46e5, #7c3aed);}
        .flashcard.back  {background: linear-gradient(135deg, #059669, #10b981);}
        .flashcard .tag {
            font-size: 0.8rem; letter-spacing: 2px; font-weight: 700;
            opacity: 0.85; margin-bottom: 0.8rem;
        }
        .flashcard .text {font-size: 1.4rem; font-weight: 600; line-height: 1.5;}
        .score-circle {
            text-align: center; font-size: 3rem; font-weight: 800;
            color: #4f46e5; margin: 0.2rem 0;
        }
        .feedback-box {
            background: linear-gradient(135deg, #eef2ff, #fdf2f8);
            border-radius: 16px; padding: 1.2rem 1.5rem; text-align: center;
            font-size: 1.1rem; color: #1f2937; border: 1px solid #e0e7ff;
            margin: 1rem 0;
        }
        div.stButton > button {border-radius: 10px; font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "page": "Home",
        "topic": "",
        "notes": "",
        "quiz": [],  # list of question dicts
        "quiz_id": 0,  # changes on restart so radio widgets reset
        "q_index": 0,
        "answers": {},  # {question_index: chosen_letter}
        "quiz_finished": False,
        "quiz_difficulty": "Medium",
        "warning": "",
        "upload_msg": None,  # (type, text) message shown after a file upload
        "flashcards": [],
        "card_index": 0,
        "card_revealed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to(page):
    st.session_state.page = page


def reset_quiz_progress():
    """Restart the current quiz from question 1 with no answers."""
    st.session_state.q_index = 0
    st.session_state.answers = {}
    st.session_state.quiz_finished = False
    st.session_state.warning = ""
    st.session_state.quiz_id += 1


# ---------------------------------------------------------------------------
# Groq API helpers
# ---------------------------------------------------------------------------
def get_api_key():
    """Order of lookup: sidebar input -> environment variable -> Streamlit secrets."""
    typed = st.session_state.get("api_key_input", "").strip()
    if typed:
        return typed
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


def friendly_api_error(error):
    """Turn Groq / network exceptions into messages a student can understand."""
    if isinstance(error, AuthenticationError):
        return "Your Groq API key is invalid or expired. Please check the key and try again."
    if isinstance(error, RateLimitError):
        return "Groq rate limit reached. Please wait a minute and try again."
    if isinstance(error, APIConnectionError):
        return "Could not connect to Groq. Please check your internet connection and try again."
    if isinstance(error, BadRequestError):
        return "Groq could not process this request. Try shorter notes or a different model."
    if isinstance(error, APIStatusError):
        return f"Groq returned an error (status {error.status_code}). Please try again shortly."
    return f"Something unexpected went wrong: {error}"


def extract_json(text):
    """Parse JSON from the model output, tolerating extra text or code fences."""
    if not text or not text.strip():
        raise ValueError("The AI returned an empty response.")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError("The AI response was not valid JSON.")


def clean_option(text):
    """Remove a leading 'A)' / 'B.' style label if the model added one."""
    return re.sub(r"^\s*[A-Da-d][\)\.\:]\s+", "", str(text)).strip()


def validate_questions(data):
    """Check the AI output and return a clean list of MCQ dictionaries."""
    items = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("The AI response did not contain a list of questions.")

    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        raw_options = item.get("options")

        if isinstance(raw_options, dict):
            options = [clean_option(raw_options.get(letter, "")) for letter in LETTERS]
        elif isinstance(raw_options, list) and len(raw_options) == 4:
            options = [clean_option(opt) for opt in raw_options]
        else:
            continue

        if not question or any(not opt for opt in options) or len(set(options)) < 4:
            continue

        answer = str(item.get("answer", "")).strip()
        letter = answer.upper()[:1]
        if letter not in LETTERS or (len(answer) > 1 and not re.match(r"^[A-Da-d][\)\.\:\s]", answer)):
            # The model may have returned the answer text instead of a letter.
            matches = [i for i, opt in enumerate(options) if opt.lower() == clean_option(answer).lower()]
            if len(matches) != 1:
                continue
            letter = LETTERS[matches[0]]

        explanation = str(item.get("explanation", "")).strip() or "No explanation provided."
        valid.append(
            {"question": question, "options": options, "answer": letter, "explanation": explanation}
        )

    if not valid:
        raise ValueError("The AI did not return any usable questions.")
    return valid


def validate_flashcards(data):
    """Check the AI output and return a clean list of flashcard dictionaries."""
    items = data.get("flashcards") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("The AI response did not contain a list of flashcards.")

    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()
        if front and back:
            valid.append({"front": front, "back": back})

    if not valid:
        raise ValueError("The AI did not return any usable flashcards.")
    return valid


def ask_groq(api_key, model, system_prompt, user_prompt, validator, attempts=2):
    """
    Call Groq and validate the result. If the AI gives a badly formatted
    answer, try again (up to `attempts` times). API errors are raised to the caller.
    """
    client = Groq(api_key=api_key, timeout=60.0)
    last_error = None
    for _ in range(attempts):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content if response.choices else ""
        try:
            return validator(extract_json(content))
        except ValueError as err:
            last_error = err
    raise ValueError(f"{last_error} Please try again.")


def generate_quiz(api_key, model, topic, notes, difficulty, num_questions):
    system_prompt = (
        "You are an expert teacher who writes exam questions. "
        "You ONLY use information found in the lesson notes given by the student. "
        "You always reply with valid JSON and nothing else."
    )
    user_prompt = f"""Topic: {topic or "Not specified"}

LESSON NOTES:
\"\"\"
{notes}
\"\"\"

Create exactly {num_questions} multiple-choice questions from the notes above.
Difficulty: {difficulty}. {DIFFICULTY_HINTS[difficulty]}

Rules:
- Base every question strictly on the lesson notes. Do not add outside facts.
- Each question has exactly four options labelled A, B, C and D.
- Exactly one option is correct. Spread the correct answers across A, B, C and D.
- Give a clear 1-2 sentence explanation of why the correct answer is right.
- Do not repeat questions.

Return ONLY JSON in this exact format:
{{
  "questions": [
    {{
      "question": "Question text?",
      "options": {{"A": "option text", "B": "option text", "C": "option text", "D": "option text"}},
      "answer": "B",
      "explanation": "Why B is correct."
    }}
  ]
}}"""
    questions = ask_groq(api_key, model, system_prompt, user_prompt, validate_questions)
    return questions[:num_questions]


def generate_flashcards(api_key, model, topic, notes, num_cards):
    system_prompt = (
        "You are an expert teacher who creates study flashcards. "
        "You ONLY use information found in the lesson notes given by the student. "
        "You always reply with valid JSON and nothing else."
    )
    user_prompt = f"""Topic: {topic or "Not specified"}

LESSON NOTES:
\"\"\"
{notes}
\"\"\"

Create exactly {num_cards} flashcards from the notes above.
- "front" is a short question or an important concept/term.
- "back" is a clear, concise answer or explanation (1-3 sentences).
- Use only information from the notes and cover different parts of them.

Return ONLY JSON in this exact format:
{{
  "flashcards": [
    {{"front": "Question or concept", "back": "Answer or explanation"}}
  ]
}}"""
    cards = ask_groq(api_key, model, system_prompt, user_prompt, validate_flashcards)
    return cards[:num_cards]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def calculate_score():
    quiz = st.session_state.quiz
    answers = st.session_state.answers
    total = len(quiz)
    correct = sum(1 for i, q in enumerate(quiz) if answers.get(i) == q["answer"])
    incorrect = total - correct
    percentage = round((correct / total) * 100) if total else 0
    return total, correct, incorrect, percentage


def feedback_message(percentage):
    if percentage == 100:
        return "🌟 Perfect score! Outstanding work, you have mastered this lesson!"
    if percentage >= 80:
        return "🎉 Excellent job! You clearly understand this material. Keep it up!"
    if percentage >= 60:
        return "👍 Good effort! You are on the right track. Review the questions you missed to improve."
    if percentage >= 40:
        return "💪 Nice try! A little more practice with the flashcards will help you improve."
    return "🌱 Don't be discouraged! Every expert started here. Review your notes, study the flashcards and try again."


# ---------------------------------------------------------------------------
# Callbacks (run before the page re-renders)
# ---------------------------------------------------------------------------
def submit_answer():
    idx = st.session_state.q_index
    choice = st.session_state.get(f"q_{st.session_state.quiz_id}_{idx}")
    if choice is None:
        st.session_state.warning = "Please select an answer before submitting."
        return
    if idx in st.session_state.answers:  # already submitted: ignore
        return
    st.session_state.answers[idx] = choice
    st.session_state.warning = ""


def next_question():
    if st.session_state.q_index < len(st.session_state.quiz) - 1:
        st.session_state.q_index += 1
    st.session_state.warning = ""


def finish_quiz():
    st.session_state.quiz_finished = True
    st.session_state.page = "Quiz Results"


def load_uploaded_file():
    """Read an uploaded .txt / .md / .pdf file and put its text in the notes box."""
    uploaded = st.session_state.get("notes_file")
    if uploaded is None:
        return

    try:
        name = uploaded.name.lower()
        if name.endswith((".txt", ".md")):
            text = uploaded.getvalue().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            if PdfReader is None:
                st.session_state.upload_msg = (
                    "error",
                    "PDF support needs the 'pypdf' package. Add it to requirements.txt and reboot the app.",
                )
                return
            reader = PdfReader(uploaded)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            st.session_state.upload_msg = ("error", "Unsupported file type. Please upload a .txt, .md or .pdf file.")
            return

        text = text.strip()
        if not text:
            st.session_state.upload_msg = (
                "error",
                "No readable text was found in this file. Scanned PDFs (images) cannot be read.",
            )
            return

        st.session_state.notes = text
        st.session_state.upload_msg = ("success", f"Loaded '{uploaded.name}' ({len(text):,} characters) into the notes box.")
    except Exception:
        st.session_state.upload_msg = ("error", "Sorry, this file could not be read. Try another file or paste the text instead.")


def prev_card():
    if st.session_state.card_index > 0:
        st.session_state.card_index -= 1
    st.session_state.card_revealed = False


def next_card():
    if st.session_state.card_index < len(st.session_state.flashcards) - 1:
        st.session_state.card_index += 1
    st.session_state.card_revealed = False


def toggle_card():
    st.session_state.card_revealed = not st.session_state.card_revealed


def restart_quiz():
    reset_quiz_progress()
    st.session_state.page = "Generate Quiz"


def clear_quiz():
    st.session_state.quiz = []
    reset_quiz_progress()


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎓 Study Buddy")
        st.caption("AI Quiz & Flashcard Generator")
        st.divider()
        st.markdown("### 🔑 Groq Settings")
        st.text_input(
            "Groq API Key",
            type="password",
            key="api_key_input",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com. You can also set the GROQ_API_KEY environment variable.",
        )
        st.selectbox("AI Model", MODELS, key="model_choice")
        if get_api_key():
            st.success("API key detected ✅")
        else:
            st.warning("No API key yet. Add one to generate content.")
        st.divider()
        st.caption("Your notes are sent to Groq only when you click a generate button.")


def render_navigation():
    cols = st.columns(len(PAGES))
    for col, page in zip(cols, PAGES):
        is_active = st.session_state.page == page
        col.button(
            f"{PAGE_ICONS[page]} {page}",
            key=f"nav_{page}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=go_to,
            args=(page,),
        )
    st.write("")


def render_home():
    st.markdown(
        """
        <div class="hero">
            <h1>🎓 AI Quiz & Flashcard Generator</h1>
            <p>Welcome, student! Paste your lesson notes and let AI turn them into
            practice quizzes and flashcards, so you can study smarter, not harder.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        """<div class="feature-card"><h3>📝 Smart Quizzes</h3>
        <p>Multiple-choice questions from your own notes, with your choice of difficulty
        and instant explanations.</p></div>""",
        unsafe_allow_html=True,
    )
    c2.markdown(
        """<div class="feature-card"><h3>🃏 Flashcards</h3>
        <p>Flip through key concepts and answers. Reveal, go back and repeat until it sticks.</p></div>""",
        unsafe_allow_html=True,
    )
    c3.markdown(
        """<div class="feature-card"><h3>🏆 Progress Tracking</h3>
        <p>See your score, correct and incorrect answers, and get encouraging feedback.</p></div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("How it works")
    st.markdown(
        """
        1. **Add your API key** in the sidebar (free from console.groq.com).
        2. Go to **Generate Quiz**, enter a title and paste your lesson notes.
        3. Choose the difficulty and number of questions, then generate your quiz.
        4. Create **Flashcards** from the same notes to review key ideas.
        5. Check your **Quiz Results** and try again to improve!
        """
    )
    st.button(
        "🚀 Get Started",
        type="primary",
        on_click=go_to,
        args=("Generate Quiz",),
    )


def render_generate_page():
    st.header("📝 Generate Quiz")
    st.write("Paste your lesson notes below and choose your quiz settings.")

    with st.container(border=True):
        st.text_input(
            "📖 Topic / Lesson title",
            key="topic",
            placeholder="e.g. Photosynthesis, World War II, Python Loops",
        )
        st.text_area(
            "🗒️ Lesson notes",
            key="notes",
            height=260,
            placeholder="Paste your lesson notes here...",
        )

        col1, col2 = st.columns(2)
        difficulty = col1.selectbox(
            "🎯 Difficulty", ["Easy", "Medium", "Hard"], index=1, key="difficulty_choice"
        )
        num_questions = col2.selectbox("🔢 Number of questions", [5, 10, 15], key="num_questions_choice")

        b1, b2 = st.columns(2)
        quiz_clicked = b1.button("✨ Generate Quiz", type="primary", use_container_width=True)
        cards_clicked = b2.button("🃏 Generate Flashcards", use_container_width=True)

    if quiz_clicked or cards_clicked:
        handle_generation(quiz_clicked, difficulty, num_questions)

    if st.session_state.quiz and not st.session_state.quiz_finished:
        st.divider()
        render_quiz()
    elif st.session_state.quiz and st.session_state.quiz_finished:
        st.divider()
        st.success("You have finished this quiz! Check the Quiz Results tab to see how you did.")
        st.button("📊 View Results", on_click=go_to, args=("Quiz Results",))


def handle_generation(quiz_clicked, difficulty, num_questions):
    """Validate input, call Groq and store the result in session state."""
    notes = st.session_state.notes.strip()
    topic = st.session_state.topic.strip()
    api_key = get_api_key()

    # --- Input validation ---
    if not notes:
        st.error("⚠️ Please paste your lesson notes before generating. The notes cannot be empty.")
        return
    if len(notes) < 40:
        st.error("⚠️ Your notes are too short to create good questions. Please add more detail.")
        return
    if not api_key:
        st.error("🔑 Please enter your Groq API key in the sidebar first.")
        return
    if len(notes) > MAX_NOTES_CHARS:
        notes = notes[:MAX_NOTES_CHARS]
        st.warning(f"Your notes were long, so only the first {MAX_NOTES_CHARS:,} characters were used.")

    model = st.session_state.get("model_choice", MODELS[0])

    try:
        if quiz_clicked:
            with st.spinner("🧠 AI is writing your quiz..."):
                questions = generate_quiz(api_key, model, topic, notes, difficulty, num_questions)
            st.session_state.quiz = questions
            st.session_state.quiz_difficulty = difficulty
            reset_quiz_progress()
            if len(questions) < num_questions:
                st.warning(f"The AI created {len(questions)} usable questions instead of {num_questions}.")
            else:
                st.success(f"✅ Your {len(questions)}-question quiz is ready!")
            st.rerun()
        else:
            with st.spinner("🃏 AI is creating your flashcards..."):
                cards = generate_flashcards(api_key, model, topic, notes, num_cards=num_questions)
            st.session_state.flashcards = cards
            st.session_state.card_index = 0
            st.session_state.card_revealed = False
            st.session_state.page = "Flashcards"
            st.rerun()
    except ValueError as err:
        st.error(f"🤖 The AI gave an unexpected response. {err}")
    except Exception as err:  # Groq errors, connection failures, anything else
        st.error(f"❌ {friendly_api_error(err)}")


def render_quiz():
    quiz = st.session_state.quiz
    idx = st.session_state.q_index
    question = quiz[idx]
    answered = idx in st.session_state.answers
    selected = st.session_state.answers.get(idx)

    st.subheader(f"Quiz: {st.session_state.topic.strip() or 'Your Lesson'}")
    st.caption(f"Difficulty: {st.session_state.quiz_difficulty}")
    st.progress((idx + (1 if answered else 0)) / len(quiz))
    st.markdown(f"**Question {idx + 1} of {len(quiz)}**")
    st.markdown(
        f'<div class="question-card">{html.escape(question["question"])}</div>',
        unsafe_allow_html=True,
    )

    st.radio(
        "Choose your answer:",
        LETTERS,
        index=None,
        format_func=lambda letter: f"{letter}.  {question['options'][LETTERS.index(letter)]}",
        key=f"q_{st.session_state.quiz_id}_{idx}",
        disabled=answered,
    )

    if st.session_state.warning and not answered:
        st.warning(st.session_state.warning)

    if answered:
        correct_letter = question["answer"]
        correct_text = question["options"][LETTERS.index(correct_letter)]
        if selected == correct_letter:
            st.success("✅ Correct! Well done.")
        else:
            st.error(f"❌ Incorrect. The correct answer is **{correct_letter}. {correct_text}**")
        st.info(f"💡 **Explanation:** {question['explanation']}")

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.button(
        "✔️ Submit Answer",
        type="primary",
        disabled=answered,
        on_click=submit_answer,
        use_container_width=True,
    )
    is_last = idx == len(quiz) - 1
    if is_last:
        c2.button(
            "🏁 Finish Quiz",
            type="primary",
            disabled=not answered,
            on_click=finish_quiz,
            use_container_width=True,
        )
    else:
        c2.button(
            "➡️ Next Question",
            disabled=not answered,
            on_click=next_question,
            use_container_width=True,
        )
    c3.button("🗑️ Discard Quiz", on_click=clear_quiz, use_container_width=True)


def render_flashcards():
    st.header("🃏 Flashcards")
    cards = st.session_state.flashcards

    if not cards:
        st.info("You have not generated any flashcards yet. Add your notes first, then click **Generate Flashcards**.")
        st.button("📝 Go to Generate Quiz", type="primary", on_click=go_to, args=("Generate Quiz",))
        return

    idx = st.session_state.card_index
    card = cards[idx]
    revealed = st.session_state.card_revealed

    st.caption(f"Card {idx + 1} of {len(cards)}")
    st.progress((idx + 1) / len(cards))

    side = "back" if revealed else "front"
    label = "ANSWER" if revealed else "QUESTION / CONCEPT"
    text = card["back"] if revealed else card["front"]
    st.markdown(
        f"""
        <div class="flashcard {side}">
            <div class="tag">{label}</div>
            <div class="text">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.button(
        "⬅️ Previous",
        disabled=idx == 0,
        on_click=prev_card,
        use_container_width=True,
    )
    c2.button(
        "🙈 Hide Answer" if revealed else "👀 Reveal Answer",
        type="primary",
        on_click=toggle_card,
        use_container_width=True,
    )
    c3.button(
        "Next ➡️",
        disabled=idx == len(cards) - 1,
        on_click=next_card,
        use_container_width=True,
    )


def render_results():
    st.header("🏆 Quiz Results")
    quiz = st.session_state.quiz

    if not quiz:
        st.info("No quiz yet. Generate a quiz and complete it to see your results here.")
        st.button("📝 Generate Quiz", type="primary", on_click=go_to, args=("Generate Quiz",))
        return

    if not st.session_state.quiz_finished:
        answered_count = len(st.session_state.answers)
        st.info(
            f"Your quiz is still in progress ({answered_count} of {len(quiz)} answered). "
            "Finish it to see your final result."
        )
        st.button("▶️ Continue Quiz", type="primary", on_click=go_to, args=("Generate Quiz",))
        return

    total, correct, incorrect, percentage = calculate_score()

    st.markdown(f'<div class="score-circle">{percentage}%</div>', unsafe_allow_html=True)
    st.progress(percentage / 100)

    m1, m2, m3 = st.columns(3)
    m1.metric("📋 Total Questions", total)
    m2.metric("✅ Correct", correct)
    m3.metric("❌ Incorrect", incorrect)

    st.markdown(f'<div class="feedback-box">{feedback_message(percentage)}</div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    b1.button("🔄 Restart Quiz", type="primary", on_click=restart_quiz, use_container_width=True)
    b2.button("🃏 Study Flashcards", on_click=go_to, args=("Flashcards",), use_container_width=True)

    st.subheader("Review your answers")
    for i, q in enumerate(quiz):
        chosen = st.session_state.answers.get(i)
        is_correct = chosen == q["answer"]
        icon = "✅" if is_correct else "❌"
        with st.expander(f"{icon} Question {i + 1}: {q['question']}"):
            for letter, option in zip(LETTERS, q["options"]):
                marker = ""
                if letter == q["answer"]:
                    marker = " ✔️ (correct answer)"
                elif letter == chosen:
                    marker = " ✖️ (your answer)"
                st.write(f"**{letter}.** {option}{marker}")
            st.info(f"💡 {q['explanation']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_state()
    inject_css()
    render_sidebar()
    render_navigation()

    page = st.session_state.page
    if page == "Home":
        render_home()
    elif page == "Generate Quiz":
        render_generate_page()
    elif page == "Flashcards":
        render_flashcards()
    elif page == "Quiz Results":
        render_results()


if __name__ == "__main__":
    main()

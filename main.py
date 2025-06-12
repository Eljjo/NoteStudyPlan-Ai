import streamlit as st
import fitz  # pymupdf
import wikipedia
import os
import google.generativeai as genai
from dotenv import load_dotenv
from fpdf import FPDF
import random
import re
import logging
from typing import List, Dict, Tuple, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY nie został znaleziony w zmiennych środowiskowych!")
        st.stop()
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
except Exception as e:
    st.error(f"❌ Błąd konfiguracji API: {str(e)}")
    st.stop()

color_emojis = ["🔵", "🟢", "🟡", "🟣", "🟠", "🔴"]


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'study_plan' not in st.session_state:
        st.session_state.study_plan = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'


def apply_theme(theme):
    """Apply custom theme styles"""
    if theme == 'dark':
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #262730;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            color: #ffffff;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b !important;
            color: #ffffff !important;
        }
        .stExpander {
            background-color: #262730;
            border: 1px solid #404040;
        }
        .stTextArea textarea {
            background-color: #262730;
            color: #ffffff;
            border: 1px solid #404040;
        }
        .stSelectbox div[data-baseweb="select"] {
            background-color: #262730;
        }
        .quiz-container {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .topic-item {
            background-color: #2d2d2d;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 4px solid #ff4b4b;
        }
        </style>
        """, unsafe_allow_html=True)

    elif theme == 'green':
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5530 100%);
            color: #e8f5e8;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1a4d1a;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1a4d1a;
            color: #e8f5e8;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50 !important;
            color: #ffffff !important;
        }
        .stExpander {
            background-color: #1a4d1a;
            border: 1px solid #4CAF50;
        }
        .stTextArea textarea {
            background-color: #1a4d1a;
            color: #e8f5e8;
            border: 1px solid #4CAF50;
        }
        .stSelectbox div[data-baseweb="select"] {
            background-color: #1a4d1a;
        }
        .quiz-container {
            background: linear-gradient(135deg, #1a4d1a 0%, #2d5a2d 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #4CAF50;
        }
        .topic-item {
            background: linear-gradient(135deg, #2d5a2d 0%, #1a4d1a 100%);
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        .sidebar .stSelectbox {
            background-color: #1a4d1a;
        }
        </style>
        """, unsafe_allow_html=True)

    elif theme == 'blue':
        st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                color: #0d47a1;
            }
            .stTabs [data-baseweb="tab-list"],
            .stTabs [data-baseweb="tab"],
            .stExpander,
            .stSelectbox div[data-baseweb="select"],
            .stTextArea textarea {
                background-color: #bbdefb;
                color: #0d47a1;
                border: 1px solid #90caf9;
            }
            .stTabs [aria-selected="true"] {
                background-color: #2196f3 !important;
                color: #ffffff !important;
            }
            .quiz-container {
                background-color: #bbdefb;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }
            .topic-item {
                background-color: #e3f2fd;
                padding: 10px;
                margin: 5px 0;
                border-radius: 8px;
                border-left: 4px solid #2196f3;
            }
            </style>
            """, unsafe_allow_html=True)

    elif theme == 'violet':
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f3e5f5 0%, #ce93d8 100%);
            color: #4a148c;
        }
        .stTabs [data-baseweb="tab-list"],
        .stTabs [data-baseweb="tab"],
        .stExpander,
        .stSelectbox div[data-baseweb="select"],
        .stTextArea textarea {
            background-color: #ce93d8;
            color: #4a148c;
            border: 1px solid #ba68c8;
        }
        .stTabs [aria-selected="true"] {
            background-color: #8e24aa !important;
            color: #ffffff !important;
        }
        .quiz-container {
            background-color: #e1bee7;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .topic-item {
            background-color: #f3e5f5;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 4px solid #8e24aa;
        }
        </style>
        """, unsafe_allow_html=True)

    elif theme == 'orange':
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            color: #e65100;
        }
        .stTabs [data-baseweb="tab-list"],
        .stTabs [data-baseweb="tab"],
        .stExpander,
        .stSelectbox div[data-baseweb="select"],
        .stTextArea textarea {
            background-color: #ffe0b2;
            color: #e65100;
            border: 1px solid #ffb74d;
        }
        .stTabs [aria-selected="true"] {
            background-color: #fb8c00 !important;
            color: #ffffff !important;
        }
        .quiz-container {
            background-color: #ffe0b2;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .topic-item {
            background-color: #fff3e0;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 4px solid #fb8c00;
        }
        </style>
        """, unsafe_allow_html=True)



    else:  # light theme
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f0f2f6;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            color: #000000;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b !important;
            color: #ffffff !important;
        }
        .quiz-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #dee2e6;
        }
        .topic-item {
            background-color: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 4px solid #ff4b4b;
        }
        </style>
        """, unsafe_allow_html=True)


def handle_api_error(func, *args, max_retries=3, **kwargs):
    """Generic API error handler with retry logic"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff


def safe_pdf_extract(file) -> Optional[str]:
    """Safely extract text from PDF with error handling"""
    try:
        if file is None:
            return None

        file_bytes = file.read()
        if not file_bytes:
            st.warning(f"⚠️ Plik {file.name} jest pusty")
            return None

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        doc.close()

        if not text.strip():
            st.warning(f"⚠️ Nie udało się wyodrębnić tekstu z {file.name}")
            return None

        return text
    except Exception as e:
        st.error(f"❌ Błąd przy czytaniu pliku {file.name}: {str(e)}")
        return None


def safe_wikipedia_search(keywords: List[str]) -> str:
    """Safely search Wikipedia with error handling"""
    summaries = []
    successful_searches = 0

    for keyword in keywords:
        try:
            # Set language to Polish
            wikipedia.set_lang("pl")
            summary = wikipedia.summary(keyword, sentences=2, auto_suggest=True, redirect=True)
            summaries.append(f"{keyword}: {summary}")
            successful_searches += 1
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                # Try with the first suggestion
                summary = wikipedia.summary(e.options[0], sentences=2)
                summaries.append(f"{keyword} ({e.options[0]}): {summary}")
                successful_searches += 1
            except:
                logger.warning(f"Wikipedia disambiguation failed for: {keyword}")
        except wikipedia.exceptions.PageError:
            logger.warning(f"Wikipedia page not found for: {keyword}")
        except Exception as e:
            logger.warning(f"Wikipedia search failed for {keyword}: {str(e)}")

    if successful_searches == 0:
        st.warning("⚠️ Nie udało się pobrać informacji z Wikipedii")
    else:
        st.info(f"✅ Pobrano informacje z Wikipedii dla {successful_searches}/{len(keywords)} słów kluczowych")

    return "\n".join(summaries)


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text with error handling"""
    try:
        if not text or not text.strip():
            return []

        lines = text.split("\n")
        keywords = set()

        for line in lines:
            cleaned_line = line.strip()
            if 3 < len(cleaned_line) <= 50 and len(cleaned_line.split()) <= 5:
                # Remove special characters and numbers
                cleaned_line = re.sub(r'[^\w\s]', '', cleaned_line)
                if cleaned_line and not cleaned_line.isdigit():
                    keywords.add(cleaned_line)

        return list(keywords)[:10]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []


def safe_ai_request(prompt: str, context: str = "general") -> Optional[str]:
    """Safely make AI requests with error handling"""
    try:
        if not prompt or not prompt.strip():
            st.error("❌ Pusty prompt do AI")
            return None

        def make_request():
            response = model.generate_content(prompt)
            return response.text

        result = handle_api_error(make_request)

        if not result or not result.strip():
            st.warning(f"⚠️ AI zwróciło pustą odpowiedź dla kontekstu: {context}")
            return None

        return result
    except Exception as e:
        st.error(f"❌ Błąd AI dla kontekstu {context}: {str(e)}")
        return None


def agent_lecture_analysis(lecture_text: str, wiki_knowledge: str) -> Optional[str]:
    """Analyze lecture with error handling"""
    if not lecture_text:
        return None

    prompt = f"""
    Przeanalizuj poniższy wykład:
    {lecture_text[:4000]}

    Wiedza uzupełniająca z Wikipedii:
    {wiki_knowledge}

    Wypisz najważniejsze tematy, trudne pojęcia, oraz krótkie streszczenie wykładu.
    """
    return safe_ai_request(prompt, "lecture_analysis")


def agent_notes_analysis(notes_text: str) -> Optional[str]:
    """Analyze notes with error handling"""
    if not notes_text:
        return None

    prompt = f"""
    Przeanalizuj poniższe notatki studenta:
    {notes_text[:4000]}

    Zidentyfikuj istotne fragmenty, pytania, które mogą wynikać z niejasności, oraz brakujące informacje.
    """
    return safe_ai_request(prompt, "notes_analysis")


def agent_final_synthesis(lecture_insights: str, notes_insight: str, wiki_knowledge: str) -> Optional[str]:
    """Create final synthesis with error handling"""
    prompt = f"""
    Na podstawie analizy wykładów:
    {lecture_insights}

    Oraz analizy notatek studenta:
    {notes_insight}

    I wiedzy z Wikipedii:
    {wiki_knowledge}

    Wygeneruj w następującym formacie:

    ## ABSTRAKT
    [Streszczenie wykładów]

    ## GŁÓWNE TEMATY
    - Temat 1
    - Temat 2
    - Temat 3

    ## PLAN WYKŁADU
    [Sekcje wykładu]

    ## QUIZ
    Stwórz quiz o odpowiedniej liczbie pytań (minimum 3, maksimum 10) dopasowanej do złożoności materiału. 
    Dla każdego pytania podaj 4 opcje odpowiedzi, oznaczając poprawną jako "(poprawna)".

    Przykład formatu:
    1. Pytanie dotyczące głównego tematu?
    - Odpowiedź A
    - Odpowiedź B (poprawna)
    - Odpowiedź C
    - Odpowiedź D

    2. Pytanie dotyczące szczegółów?
    - Odpowiedź A (poprawna)
    - Odpowiedź B
    - Odpowiedź C
    - Odpowiedź D

    [Kontynuuj z odpowiednią liczbą pytań...]

    ## MATERIAŁY EDUKACYJNE
    [Materiały dla każdego tematu - 2-3 zdania wyjaśniające każdy główny temat]
    """
    return safe_ai_request(prompt, "final_synthesis")


def parse_quiz_from_text(text: str) -> List[Dict]:
    """Parse quiz questions from AI output with better error handling"""
    try:
        quiz_questions = []

        # Find the quiz section
        quiz_section = ""
        lines = text.split('\n')
        in_quiz_section = False

        for line in lines:
            if "## QUIZ" in line.upper() or "QUIZ" in line and line.strip().startswith('#'):
                in_quiz_section = True
                continue
            elif line.strip().startswith('##') and in_quiz_section:
                break
            elif in_quiz_section:
                quiz_section += line + '\n'

        if not quiz_section:
            # Fallback: look for numbered questions anywhere in text
            quiz_section = text

        # Parse questions and answers
        question_pattern = r'(\d+)\.\s*([^?]+\?)\s*\n((?:\s*-\s*[^\n]+\n?)+)'
        matches = re.findall(question_pattern, quiz_section, re.MULTILINE)

        for match in matches:
            question_num, question_text, answers_text = match

            # Parse answers
            answer_lines = [line.strip() for line in answers_text.split('\n') if line.strip().startswith('-')]
            answers = []
            correct_answer = None

            for answer_line in answer_lines:
                answer = answer_line.replace('-', '').strip()
                if '(poprawna)' in answer.lower():
                    answer = answer.replace('(poprawna)', '').replace('(Poprawna)', '').strip()
                    correct_answer = answer
                answers.append(answer)

            if len(answers) >= 2:  # At least 2 answers required
                quiz_questions.append({
                    'question': question_text.strip(),
                    'answers': answers,
                    'correct': correct_answer or answers[0]  # Default to first if no correct marked
                })

        return quiz_questions
    except Exception as e:
        logger.error(f"Error parsing quiz: {str(e)}")
        return []


def safe_pdf_export(content: str, filename: str = "plan_nauki.pdf") -> Optional[str]:
    """Safely export content to PDF"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Handle UTF-8 encoding issues
        for line in content.strip().split("\n"):
            try:
                # Remove non-ASCII characters that might cause issues
                cleaned_line = line.encode('latin-1', 'ignore').decode('latin-1')
                pdf.multi_cell(0, 10, txt=cleaned_line)
            except:
                # Skip problematic lines
                continue

        pdf_path = f"/tmp/{filename}"
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        st.error(f"❌ Błąd przy eksporcie PDF: {str(e)}")
        return None


def parse_ai_output_sections(ai_output: str) -> Dict[str, str]:
    """Parse AI output into separate sections"""
    sections = {
        'abstract': '',
        'topics': [],
        'plan': '',
        'quiz': '',
        'materials': ''
    }

    try:
        lines = ai_output.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            line_upper = line.strip().upper()

            if '## ABSTRAKT' in line_upper:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'abstract'
                current_content = []
            elif '## GŁÓWNE TEMATY' in line_upper or '## TEMATY' in line_upper:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'topics'
                current_content = []
            elif '## PLAN WYKŁADU' in line_upper or '## PLAN' in line_upper:
                if current_section:
                    if current_section == 'topics':
                        # Extract topic list
                        topic_lines = [l.strip().lstrip('- ') for l in current_content if l.strip().startswith('-')]
                        sections['topics'] = topic_lines
                    else:
                        sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'plan'
                current_content = []
            elif '## QUIZ' in line_upper:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'quiz'
                current_content = []
            elif '## MATERIAŁY' in line_upper:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'materials'
                current_content = []
            elif current_section and not line.strip().startswith('##'):
                current_content.append(line)

        # Don't forget the last section
        if current_section:
            if current_section == 'topics':
                topic_lines = [l.strip().lstrip('- ') for l in current_content if l.strip().startswith('-')]
                sections['topics'] = topic_lines
            else:
                sections[current_section] = '\n'.join(current_content).strip()

    except Exception as e:
        logger.error(f"Error parsing AI output sections: {str(e)}")

    return sections


def display_analysis_results(ai_output: str):
    """Display analysis results in organized sections with theme support"""
    sections = parse_ai_output_sections(ai_output)

    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Abstrakt", "📚 Materiały", "📝 Plan wykładu", "🎯 Pełny tekst"])

    with tab1:
        st.header("📋 Abstrakt")
        if sections['abstract']:
            st.markdown(sections['abstract'])
        else:
            st.info("Abstrakt nie został wygenerowany lub nie został rozpoznany.")

        st.subheader("🎯 Główne tematy")
        if sections['topics']:
            for i, topic in enumerate(sections['topics'], 1):
                if topic.strip():
                    emoji = random.choice(color_emojis)
                    st.markdown(f"""
                    <div class="topic-item">
                        {emoji} <strong>{i}. {topic.strip()}</strong>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Tematy nie zostały wygenerowane lub nie zostały rozpoznane.")

    with tab2:
        st.header("📚 Materiały edukacyjne")
        if sections['materials']:
            # Try to split materials by topics if possible
            materials_text = sections['materials']

            # Check if materials are organized by topics
            if '**' in materials_text or '#' in materials_text:
                st.markdown(materials_text)
            else:
                # Simple paragraph display
                paragraphs = [p.strip() for p in materials_text.split('\n\n') if p.strip()]
                for i, paragraph in enumerate(paragraphs, 1):
                    with st.expander(f"📖 Materiał {i}", expanded=True):
                        st.markdown(paragraph)
        else:
            st.info("Materiały edukacyjne nie zostały wygenerowane.")

    with tab3:
        st.header("📝 Plan wykładu")
        if sections['plan']:
            st.markdown(sections['plan'])
        else:
            st.info("Plan wykładu nie został wygenerowany.")

    with tab4:
        st.header("🎯 Pełny tekst analizy")
        with st.expander("Pokaż pełny tekst", expanded=False):
            st.text_area("Kompletna analiza AI", ai_output, height=400, key="full_text_display")


def create_study_plan(days: int, topics: List[str]) -> str:
    """Create study plan with error handling"""
    try:
        if not topics or days <= 0:
            return "Brak tematów do planowania"

        plan = ""
        per_day = max(1, (len(topics) + days - 1) // days)

        for i in range(days):
            start = i * per_day
            end = min(start + per_day, len(topics))
            day_topics = topics[start:end]

            if day_topics:
                plan += f"Dzień {i + 1} - Tematy: {', '.join(day_topics)}\n"

        return plan
    except Exception as e:
        logger.error(f"Error creating study plan: {str(e)}")
        return f"Błąd przy tworzeniu planu: {str(e)}"


# --- MAIN UI ---
def main():
    init_session_state()

    st.title("NoteStudyPlan Al")

    # Theme selector in sidebar
    with st.sidebar:
        st.markdown("### Motyw aplikacji")
        theme_options = {
            'Ciemny': 'dark',
            'Jasny': 'light',
            'Zielony': 'green',
            'Zielony': 'green',
            'Niebieski': 'blue',
            'Fioletowy': 'violet',
            'Pomarańczowy': 'orange',
        }

        selected_theme = st.selectbox(
            "Wybierz motyw:",
            options=list(theme_options.keys()),
            index=list(theme_options.values()).index(st.session_state.theme)
        )

        if theme_options[selected_theme] != st.session_state.theme:
            st.session_state.theme = theme_options[selected_theme]
            st.rerun()

        st.markdown("---")
        st.markdown(
            "**Instrukcja:** Wgraj jeden lub więcej PDF-ów z wykładami i opcjonalnie notatek. Podaj, ile dni chcesz poświęcić na naukę.")

    # Apply selected theme
    apply_theme(st.session_state.theme)

    # File uploaders
    pdf_lectures = st.file_uploader("Wgraj PDF-y wykładowe", type="pdf", accept_multiple_files=True)
    pdf_notes = st.file_uploader("Wgraj notatki studenta (opcjonalnie)", type="pdf", accept_multiple_files=True)
    days = st.number_input("Dni na naukę", min_value=1, max_value=30, value=3)

    # Process materials
    if st.button("📄 Przetwórz materiały") and pdf_lectures:
        with st.spinner("Czytam materiały..."):
            try:
                # Reset session state
                st.session_state.processing_complete = False

                # Extract lecture texts
                lecture_texts = {}
                failed_files = []

                for file in pdf_lectures:
                    text = safe_pdf_extract(file)
                    if text:
                        lecture_texts[file.name] = text
                    else:
                        failed_files.append(file.name)

                if failed_files:
                    st.warning(f"⚠️ Nie udało się przetworzyć plików: {', '.join(failed_files)}")

                if not lecture_texts:
                    st.error("❌ Nie udało się przetworzyć żadnego pliku wykładowego!")
                    return

                # Extract keywords and get Wikipedia knowledge
                combined_text = "\n".join(lecture_texts.values())
                keywords = extract_keywords(combined_text)

                if keywords:
                    wiki_knowledge = safe_wikipedia_search(keywords)
                else:
                    wiki_knowledge = ""
                    st.warning("⚠️ Nie znaleziono słów kluczowych")

                # Analyze lectures
                lecture_insights = ""
                for fname, text in lecture_texts.items():
                    st.info(f"🧠 Analiza wykładu: {fname}...")
                    insight = agent_lecture_analysis(text, wiki_knowledge)
                    if insight:
                        lecture_insights += f"\n\n### Analiza wykładu: {fname}\n{insight}"

                # Analyze notes if provided
                notes_insight = ""
                if pdf_notes:
                    st.info("🧠 Analiza notatek studenta...")
                    notes_texts = []
                    for file in pdf_notes:
                        text = safe_pdf_extract(file)
                        if text:
                            notes_texts.append(text)

                    if notes_texts:
                        combined_notes = "\n".join(notes_texts)
                        notes_insight = agent_notes_analysis(combined_notes) or ""

                # Final synthesis
                st.info("🤝 Agenci współpracują nad materiałami edukacyjnymi...")
                ai_output = agent_final_synthesis(lecture_insights, notes_insight, wiki_knowledge)

                if not ai_output:
                    st.error("❌ Nie udało się wygenerować materiałów edukacyjnych!")
                    return

                # Parse quiz and topics
                quiz_data = parse_quiz_from_text(ai_output)

                # Extract topics using the new parser
                sections = parse_ai_output_sections(ai_output)
                topic_lines = sections.get('topics', [])

                # Fallback topic extraction if parser didn't work
                if not topic_lines:
                    lines = ai_output.split('\n')
                    in_topics_section = False

                    for line in lines:
                        if '## GŁÓWNE TEMATY' in line.upper():
                            in_topics_section = True
                            continue
                        elif line.strip().startswith('##') and in_topics_section:
                            break
                        elif in_topics_section and line.strip().startswith('-'):
                            topic = line.strip().lstrip('- ').strip()
                            if topic:
                                topic_lines.append(topic)

                # Create study plan
                study_plan = create_study_plan(days, topic_lines) if topic_lines else ""

                # Store in session state
                st.session_state.processed_data = ai_output
                st.session_state.quiz_data = quiz_data
                st.session_state.study_plan = study_plan
                st.session_state.topic_lines = topic_lines
                st.session_state.processing_complete = True

                st.success("✅ Przetwarzanie zakończone pomyślnie!")

            except Exception as e:
                st.error(f"❌ Błąd podczas przetwarzania: {str(e)}")
                logger.error(f"Processing error: {str(e)}")

    # Display results if processing is complete
    if st.session_state.processing_complete and st.session_state.processed_data:
        display_analysis_results(st.session_state.processed_data)

        # Display quiz
        if st.session_state.quiz_data:
            quiz_count = len(st.session_state.quiz_data)
            st.subheader(f"🧪 Quiz sprawdzający ({quiz_count} pytań)")

            st.markdown('<div class="quiz-container">', unsafe_allow_html=True)

            with st.form("quiz_form"):
                user_answers = []

                for i, q_data in enumerate(st.session_state.quiz_data):
                    st.write(f"**{i + 1}. {q_data['question']}**")
                    user_choice = st.radio(
                        "Wybierz odpowiedź:",
                        options=q_data['answers'],
                        key=f"quiz_q_{i}"
                    )
                    user_answers.append((q_data['question'], user_choice, q_data['correct']))

                submitted = st.form_submit_button("✅ Sprawdź odpowiedzi")

            st.markdown('</div>', unsafe_allow_html=True)

            if submitted:
                score = 0
                st.markdown("### 📊 Wynik:")

                for idx, (question, chosen, correct) in enumerate(user_answers):
                    if chosen == correct:
                        st.success(f"{idx + 1}. ✅ Poprawnie!")
                        score += 1
                    else:
                        st.error(f"{idx + 1}. ❌ Błędna odpowiedź")
                        st.write(f"*Twoja odpowiedź:* {chosen}")
                        st.write(f"*Poprawna odpowiedź:* {correct}")

                percentage = (score / len(user_answers)) * 100

                # Add performance feedback
                if percentage >= 90:
                    performance_msg = "🌟 Doskonały wynik! Świetnie opanowałeś materiał!"
                elif percentage >= 70:
                    performance_msg = "👍 Dobry wynik! Warto przejrzeć kilka tematów."
                elif percentage >= 50:
                    performance_msg = "📚 Przeciętny wynik. Poświęć więcej czasu na naukę."
                else:
                    performance_msg = "📖 Warto dokładnie przejrzeć materiał przed kolejną próbą."

                st.markdown(f"## 🏁 Twój wynik: **{score} / {len(user_answers)} ({percentage:.1f}%)**")
                st.info(performance_msg)
        else:
            st.info("🤖 Quiz nie został wygenerowany lub wystąpił błąd podczas parsowania pytań.")

        # Display study plan
        if st.session_state.study_plan and st.session_state.topic_lines:
            with st.sidebar:
                st.markdown("## 📅 Plan nauki")
                for line in st.session_state.study_plan.strip().split("\n"):
                    if "Dzień" in line:
                        st.markdown(f"### 🗓️ {line.split('-')[0].strip()}")
                        topics_str = line.split("- Tematy:")[-1].strip()
                        topics_list = [t.strip() for t in topics_str.split(",")]
                        for topic in topics_list:
                            emoji = random.choice(color_emojis)
                            st.markdown(f"- {emoji} **{topic}**")

            # PDF export
            if st.button("📥 Eksportuj plan nauki do PDF"):
                pdf_path = safe_pdf_export(st.session_state.study_plan)
                if pdf_path:
                    try:
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="📄 Pobierz PDF",
                                data=f.read(),
                                file_name="plan_nauki.pdf",
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.error(f"❌ Błąd przy pobieraniu PDF: {str(e)}")

        st.subheader("💡 Co dalej?")
        st.markdown(
            "- Przeglądaj streszczenie\n- Ucz się tematycznie wg planu\n- Wygeneruj test sprawdzający po zakończeniu")


if __name__ == "__main__":
    main()
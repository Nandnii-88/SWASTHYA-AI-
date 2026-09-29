import os
import base64
from pathlib import Path

from dotenv import load_dotenv

# Gemini
from google import genai
from google.genai import types

# OpenAI
from openai import OpenAI

# Groq
from groq import Groq


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ============================================================
# API KEYS
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# MODELS
# ============================================================

# Gemini is responsible for ACTUAL IMAGE ANALYSIS.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.0-flash"
)

# OpenAI is TEXT ONLY in this application.
OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)

# Groq is TEXT ONLY in this application.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


# ============================================================
# CLIENT INITIALIZATION
# ============================================================

gemini_client = None
openai_client = None
groq_client = None


if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )
    except Exception as e:
        print(
            "WARNING: Gemini client initialization failed:",
            repr(e)
        )


if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY
        )
    except Exception as e:
        print(
            "WARNING: OpenAI client initialization failed:",
            repr(e)
        )


if GROQ_API_KEY:
    try:
        groq_client = Groq(
            api_key=GROQ_API_KEY
        )
    except Exception as e:
        print(
            "WARNING: Groq client initialization failed:",
            repr(e)
        )


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
    "English": "English",
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Telugu": "Telugu",
    "Marathi": "Marathi",
    "Tamil": "Tamil",
    "Gujarati": "Gujarati",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Punjabi": "Punjabi",
    "Urdu": "Urdu",
    "Odia": "Odia",
    "Assamese": "Assamese",
    "Sanskrit": "Sanskrit",

    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are SWASTHYAAI, a multilingual health-awareness assistant.

You provide health information and education.

You are NOT a doctor and must not present yourself as one.

IMPORTANT SAFETY RULES:

1. Never claim certainty about a diagnosis.

2. Never say that an image proves a disease or condition.

3. When discussing symptoms or images:
   - distinguish observations from possibilities
   - explain uncertainty
   - mention limitations
   - recommend appropriate professional care

4. For images:
   - describe only what is visibly present
   - do not invent details
   - do not diagnose from visual appearance alone
   - clearly state when image quality limits assessment

5. Do not prescribe prescription medication.

6. Do not tell users to start, stop, increase,
   or decrease prescribed medication.

7. If potentially serious warning signs are present,
   recommend urgent medical attention.

8. For users in India, emergency services can be reached
   through 112.

9. Respond entirely in the requested language.

10. Be helpful, calm, clear, and concise.

11. Never reveal internal API details, model names,
    provider results, internal confidence values,
    or implementation details to the user.
"""


# ============================================================
# HELPERS
# ============================================================

def get_language(language):
    """
    Convert a language code/name into the language name.
    """

    if not language:
        return "English"

    return LANGUAGES.get(
        language,
        str(language)
    )


def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# EMERGENCY DETECTION
# ============================================================

EMERGENCY_KEYWORDS = [

    # Cardiac / breathing
    "chest pain",
    "chest pressure",
    "chest tightness",
    "difficulty breathing",
    "trouble breathing",
    "can't breathe",
    "cannot breathe",
    "shortness of breath",
    "severe breathlessness",

    # Neurological
    "stroke",
    "face drooping",
    "slurred speech",
    "can't speak",
    "cannot speak",
    "weakness on one side",
    "one side weakness",
    "paralysis",
    "seizure",
    "convulsion",
    "unconscious",
    "unresponsive",
    "passed out",
    "fainted",

    # Bleeding / trauma
    "severe bleeding",
    "heavy bleeding",
    "uncontrolled bleeding",
    "major injury",
    "severe injury",

    # Allergic reaction
    "anaphylaxis",
    "severe allergic reaction",
    "swelling of throat",
    "throat swelling",

    # Poisoning
    "overdose",
    "poisoning",
    "poison",

    # Burns
    "severe burn",

    # Mental-health emergency
    "suicidal",
    "suicide",
    "kill myself",
]


def detect_emergency(text):
    """
    Basic emergency keyword detection.

    Returns:
        True  -> potentially urgent symptom detected
        False -> no emergency keyword detected

    This is only a basic safety layer and does NOT
    replace professional medical assessment.
    """

    if not text:
        return False

    text_lower = str(text).lower()

    for keyword in EMERGENCY_KEYWORDS:

        if keyword in text_lower:
            return True

    return False


# ============================================================
# GEMINI TEXT
# ============================================================

def ask_gemini_text(
    message,
    history="",
    language="English"
):

    if gemini_client is None:

        print(
            "GEMINI TEXT: client unavailable"
        )

        return None

    language_name = get_language(
        language
    )

    prompt = f"""
{SYSTEM_PROMPT}

Requested language:
{language_name}

Conversation history:
{history}

Current user message:
{message}

Provide a useful health-awareness response.

If the user describes potentially dangerous symptoms,
clearly explain that urgent professional care may be needed.

Do not provide a definitive diagnosis.
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )

        result = clean_text(
            getattr(
                response,
                "text",
                ""
            )
        )

        if result:
            return result

        print(
            "GEMINI TEXT ERROR: empty response"
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("GEMINI TEXT ERROR")
        print("TYPE:", type(e).__name__)
        print("MESSAGE:", repr(e))
        print("=" * 70 + "\n")

    return None


# ============================================================
# OPENAI TEXT
# ============================================================

def ask_openai_text(
    message,
    history="",
    language="English"
):

    if openai_client is None:

        print(
            "OPENAI TEXT: client unavailable"
        )

        return None

    language_name = get_language(
        language
    )

    prompt = f"""
{SYSTEM_PROMPT}

Requested language:
{language_name}

Conversation history:
{history}

Current user message:
{message}

Provide a useful health-awareness response.
Do not provide a definitive diagnosis.
"""

    try:

        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            store=False
        )

        result = clean_text(
            getattr(
                response,
                "output_text",
                ""
            )
        )

        if result:
            return result

        print(
            "OPENAI TEXT ERROR: empty response"
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("OPENAI TEXT ERROR")
        print("TYPE:", type(e).__name__)
        print("MESSAGE:", repr(e))
        print("=" * 70 + "\n")

    return None


# ============================================================
# GROQ TEXT
# ============================================================

def ask_groq_text(
    message,
    history="",
    language="English"
):

    if groq_client is None:

        print(
            "GROQ TEXT: client unavailable"
        )

        return None

    language_name = get_language(
        language
    )

    prompt = f"""
Requested language:
{language_name}

Conversation history:
{history}

Current user message:
{message}

Provide a cautious health-awareness response.

Do not diagnose with certainty.

Do not prescribe prescription medication.

Explain when professional care is appropriate.
"""

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_completion_tokens=1500
        )

        result = clean_text(
            response.choices[0]
            .message
            .content
        )

        if result:
            return result

        print(
            "GROQ TEXT ERROR: empty response"
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("GROQ TEXT ERROR")
        print("TYPE:", type(e).__name__)
        print("MESSAGE:", repr(e))
        print("=" * 70 + "\n")

    return None


# ============================================================
# MAIN CHAT FUNCTION
# ============================================================

def ask_gemini(
    message,
    history="",
    language="English"
):

    language_name = get_language(
        language
    )

    print("\n")
    print("=" * 70)
    print("SWASTHYAAI 3-API TEXT VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Emergency layer
    # --------------------------------------------------------

    emergency = detect_emergency(
        message
    )

    # --------------------------------------------------------
    # Run all three providers
    # --------------------------------------------------------

    gemini_result = ask_gemini_text(
        message,
        history,
        language_name
    )

    openai_result = ask_openai_text(
        message,
        history,
        language_name
    )

    groq_result = ask_groq_text(
        message,
        history,
        language_name
    )

    print(
        "Gemini:",
        "SUCCESS" if gemini_result else "FAILED"
    )

    print(
        "OpenAI:",
        "SUCCESS" if openai_result else "FAILED"
    )

    print(
        "Groq:",
        "SUCCESS" if groq_result else "FAILED"
    )

    successful = [
        result
        for result in [
            gemini_result,
            openai_result,
            groq_result
        ]
        if result
    ]

    print(
        "Successful providers:",
        len(successful)
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Nothing worked
    # --------------------------------------------------------

    if not successful:

        print(
            "TEXT ANALYSIS FAILED COMPLETELY"
        )

        print("=" * 70)
        print()

        return (
            "I couldn't generate a response right now. "
            "Please check that the AI services and API keys "
            "are configured correctly."
        )

    # --------------------------------------------------------
    # Only one provider worked
    # --------------------------------------------------------

    if len(successful) == 1:

        answer = successful[0]

        if emergency:

            answer = (
                "⚠️ **Potential emergency**\n\n"
                "Some of the symptoms you described may "
                "require urgent medical attention. If the "
                "symptoms are severe, worsening, or life-threatening, "
                "please seek emergency care immediately. "
                "In India, you can call **112**.\n\n"
                + answer
            )

        return answer

    # --------------------------------------------------------
    # Multiple providers:
    # Use Gemini as final synthesis.
    # --------------------------------------------------------

    if gemini_client is not None:

          if gemini_client is not None:

        synthesis_prompt = f"""
{SYSTEM_PROMPT}

Requested language:
{language_name}
...

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Aplikacja do analizy tekstu i tłumaczenia")
st.write(
    """
    **Do czego służy ta aplikacja?**
    Aplikacja pozwala na przetwarzanie tekstu w języku naturalnym przy użyciu modeli AI z Hugging Face.
    Dostępne funkcje:
    - **Rozpoznawanie języka** — sprawdź w jakim języku napisany jest tekst
    - **Tłumaczenie EN → DE** — przetłumacz tekst z języka angielskiego na niemiecki
    - **Wydźwięk emocjonalny tekstu** - sprawdź wydźwięk tekstu
    """
)

st.info("Wybierz opcję z listy poniżej i wpisz tekst, który chcesz przetworzyć.")

st.success("Aplikacja uruchomiona pomyślnie!")

LANGUAGE_NAMES = {
    "ar": "Arabski", "bg": "Bułgarski", "de": "Niemiecki", "el": "Grecki",
    "en": "Angielski", "es": "Hiszpański", "fr": "Francuski", "hi": "Hindi",
    "it": "Włoski", "ja": "Japoński", "nl": "Niderlandzki", "pl": "Polski",
    "pt": "Portugalski", "ro": "Rumuński", "ru": "Rosyjski", "sw": "Suahili",
    "th": "Tajski", "tr": "Turecki", "ur": "Urdu", "vi": "Wietnamski",
    "zh": "Chiński",
}

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_language_detector():
    return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

@st.cache_resource
def load_translator():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    return tokenizer, model

st.header("Przetwarzanie języka naturalnego")

option = st.selectbox(
    "Wybierz funkcję",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Rozpoznawanie języka",
        "Tłumaczenie EN → DE",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    st.subheader("Analiza wydźwięku emocjonalnego")
    st.write("Wpisz tekst po angielsku, a model oceni czy ma wydźwięk pozytywny czy negatywny.")
    text = st.text_area(label="Wpisz tekst po angielsku")
    if text:
        if len(text.strip()) < 3:
            st.warning("Tekst jest za krótki — wpisz co najmniej kilka słów.")
        else:
            with st.spinner("Analizuję wydźwięk..."):
                try:
                    classifier = load_sentiment()
                    answer = classifier(text)
                    label = answer[0]["label"]
                    score = round(answer[0]["score"] * 100, 2)
                    if label == "POSITIVE":
                        st.success(f"Wydźwięk: POZYTYWNY (pewność: {score}%)")
                    else:
                        st.error(f"Wydźwięk: NEGATYWNY (pewność: {score}%)")
                except Exception as e:
                    st.error(f"Nie udało się przeanalizować tekstu: {e}")

elif option == "Rozpoznawanie języka":
    st.subheader("Rozpoznawanie języka")
    st.write("Wpisz dowolny tekst, a ja postaram sie odgadnac co to za jezyk :)")
    st.info("Obsługiwane języki: Arabski, Bułgarski, Niemiecki, Grecki, Angielski,"
            " Hiszpański, Francuski, Hindi, Włoski, Japoński, Niderlandzki, Polski,"
            " Portugalski, Rumuński, Rosyjski, Suahili, Tajski, Turecki, Urdu,"
            " Wietnamski, Chiński")
    text = st.text_area(label="Wpisz tekst")
    if text:
        if len(text.strip()) < 3:
            st.warning("Tekst jest za krótki — wpisz co najmniej kilka znaków.")
        else:
            with st.spinner("Rozpoznaję język... (prosze chwilke poczekac)"):
                try:
                    detector = load_language_detector()
                    result = detector(text)
                    lang = result[0]["label"]
                    score = round(result[0]["score"] * 100, 2)
                    lang_name = LANGUAGE_NAMES.get(lang, lang)
                    if score < 50:
                        st.warning(f"Wykryty język: **{lang_name}**, ale pewność jest niska ({score}%). Spróbuj wpisać dłuższy tekst.")
                    else:
                        st.success(f"Wykryty język: **{lang_name}** (pewność: {score}%)")
                except Exception:
                    st.error("Nie udało się rozpoznać języka. Sprawdź połączenie z internetem i spróbuj ponownie.")

elif option == "Tłumaczenie EN → DE":
    st.subheader("Tłumaczenie angielski → niemiecki")
    st.write("Wpisz tekst w języku angielskim, a model przetłumaczy go na język niemiecki.")
    text = st.text_area(label="Wpisz tekst po angielsku")
    if text:
        if len(text.strip()) < 2:
            st.warning("Tekst jest za krótki — wpisz co najmniej kilka słów.")
        else:
            with st.spinner("Tłumaczę tekst... (prosze chwilke poczekac)"):
                try:
                    tokenizer, model = load_translator()
                    inputs = tokenizer(text, return_tensors="pt", padding=True)
                    translated = model.generate(**inputs)
                    result = tokenizer.decode(translated[0], skip_special_tokens=True)
                    st.success("Tłumaczenie zakończone!")
                    st.write("**Wynik:**")
                    st.write(result)
                except Exception as e:
                    st.error(f"Nie udało się przetłumaczyć tekstu: {e}")

st.divider()
st.caption("Numer indeksu: s28759")

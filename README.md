# 🧠 NoteStudyPlan AI

NoteStudyPlan AI to inteligentna aplikacja edukacyjna, która analizuje Twoje materiały PDF — takie jak **skrypty z wykładów** i **notatki** — i automatycznie tworzy:

- ✍️ Streszczenie materiałów
- 📅 Personalizowany plan nauki
- 🧪 Interaktywny quiz sprawdzający wiedzę

---

## 📦 Funkcje aplikacji

✅ **Wgrywanie materiałów PDF**  
Użytkownik może przesłać wiele plików PDF z wykładów oraz opcjonalnie własne notatki.

✅ **Analiza i ekstrakcja wiedzy**  
Za pomocą AI aplikacja:
- Streszcza wykłady
- Identyfikuje główne tematy
- Łączy informacje z notatkami
- Wzbogaca kontekst o wiedzę z Wikipedii

✅ **Plan nauki na wybraną liczbę dni**  
Na podstawie tematów generowany jest dostosowany do użytkownika plan nauki podzielony na dni.

✅ **Quiz generowany automatycznie**  
Na koniec użytkownik otrzymuje quiz z pytaniami jednokrotnego wyboru, który pozwala sprawdzić poziom zrozumienia materiału.

✅ **Różne motywy interfejsu**  
Dzięki wbudowanemu systemowi motywów można dopasować wygląd aplikacji do swoich preferencji (ciemny, jasny, zielony, fioletowy itp.).

✅ **Eksport planu nauki do PDF**  
Gotowy plan nauki można pobrać jako plik PDF.

---

## 🧰 Technologie

- [Streamlit](https://streamlit.io/) – szybkie tworzenie aplikacji webowych w Pythonie
- [Google Generative AI (Gemini 1.5 Flash)](https://ai.google.dev/) – zaawansowany model językowy do analizy tekstu
- [Wikipedia API] – automatyczne wzbogacenie kontekstu o zewnętrzną wiedzę
- `PyMuPDF` (`fitz`) – ekstrakcja tekstu z plików PDF
- `fpdf` – generowanie planu nauki w formacie PDF

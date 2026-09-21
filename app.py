import streamlit as st
import joblib
import re


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

model = joblib.load(
    "model/sentiment_model.pkl"
)

tfidf = joblib.load(
    "model/tfidf_vectorizer.pkl"
)


# =========================================================
# TEXT CLEANING FUNCTION
# =========================================================

def clean_text(text):

    text = text.lower()

    # Remove HTML <br> tags
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    # Keep alphabets, apostrophes and spaces
    text = re.sub(
        r"[^a-zA-Z'\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# SENTENCE SUGGESTIONS
# =========================================================

sentence_suggestions = [

    # -----------------------------------------------------
    # POSITIVE SENTENCES
    # -----------------------------------------------------

    "The movie was really amazing",
    "The movie was absolutely fantastic",
    "The movie was very entertaining",
    "The movie was highly enjoyable",
    "The movie was a wonderful experience",

    "The story was engaging and well written",
    "The story was interesting and enjoyable",
    "The acting was excellent and convincing",
    "The performances were outstanding",
    "The direction was brilliant",
    "The characters were interesting and memorable",
    "The ending was satisfying and emotional",
    "The cinematography was beautiful",
    "The soundtrack was excellent",

    "I really enjoyed watching this movie",
    "I really liked this movie",
    "I loved this movie",
    "I would definitely recommend this movie",
    "This movie is worth watching",
    "This was one of the best movies I have watched",

    # -----------------------------------------------------
    # NEGATIVE SENTENCES
    # -----------------------------------------------------

    "The movie was very boring",
    "The movie was extremely disappointing",
    "The movie was a complete waste of time",

    "The story was weak and predictable",
    "The acting was terrible",
    "The performances were disappointing",
    "The direction was poor",
    "The characters were boring and uninteresting",
    "The ending was disappointing",
    "The plot was confusing and slow",
    "The dialogue was weak",
    "The visuals were not impressive",

    "I did not enjoy watching this movie",
    "I did not like this movie",
    "I hated this movie",
    "I would not recommend this movie",
    "This movie was not worth watching",

    # -----------------------------------------------------
    # MIXED / NEUTRAL SENTENCES
    # -----------------------------------------------------

    "The movie was okay but could be better",
    "The story was interesting but the ending was weak",
    "The acting was good but the plot was predictable",
    "The movie had some good moments but was not very engaging",
    "The movie was average",
    "Nothing special about this movie",
    "The movie could have been better"
]


# =========================================================
# WORD SUGGESTIONS
# =========================================================

word_suggestions = [

    "amazing",
    "excellent",
    "fantastic",
    "great",
    "good",
    "wonderful",
    "brilliant",
    "beautiful",
    "enjoyable",
    "entertaining",
    "interesting",
    "outstanding",
    "satisfying",
    "recommended",

    "terrible",
    "horrible",
    "boring",
    "disappointing",
    "bad",
    "poor",
    "weak",
    "predictable",
    "confusing",
    "slow",
    "waste"
]


# Remove duplicates
sentence_suggestions = list(
    dict.fromkeys(sentence_suggestions)
)

word_suggestions = list(
    dict.fromkeys(word_suggestions)
)


# =========================================================
# STREAMLIT V2 COMPONENT
# =========================================================

review_autocomplete = st.components.v2.component(

    "review_autocomplete",

    html="""
    <div class="autocomplete-container">

        <textarea
            id="reviewBox"
            rows="7"
            placeholder="Start typing your review..."
        ></textarea>

        <div
            id="suggestions"
            class="suggestions-box"
        ></div>

        <button
            id="analyzeButton"
            type="button"
        >
            🔍 Analyze Review
        </button>

    </div>
    """,

    css="""
    .autocomplete-container {
        width: 100%;
        position: relative;
        font-family: Arial, sans-serif;
    }

    #reviewBox {
        width: 100%;
        min-height: 150px;
        box-sizing: border-box;
        padding: 14px;
        font-size: 16px;
        line-height: 1.5;
        border: 1px solid #cccccc;
        border-radius: 8px;
        resize: vertical;
        outline: none;
    }

    #reviewBox:focus {
        border-color: #666666;
    }

   .suggestions-box {
    width: 100%;
    box-sizing: border-box;

    background-color: #ffffff;
    color: #222222;

    border: 1px solid #dddddd;
    border-radius: 6px;

    margin-top: 4px;

    max-height: 240px;
    overflow-y: auto;

    display: none;

    position: relative;
    z-index: 1000;

    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}


.suggestion-item {
    padding: 10px 12px;

    background-color: #ffffff;
    color: #222222;

    cursor: pointer;

    font-size: 14px;

    border-bottom: 1px solid #eeeeee;
}


.suggestion-item:hover,
.suggestion-item.active {
    background-color: #f0f0f0;
    color: #111111;
}

    #analyzeButton {
        margin-top: 14px;
        padding: 10px 18px;
        border: none;
        border-radius: 7px;
        font-size: 15px;
        cursor: pointer;
        background: #ff4b4b;
        color: white;
    }

    #analyzeButton:hover {
        background: #e63939;
    }
    """,

    js="""
    export default function(component) {

        const {
            data,
            parentElement,
            setStateValue,
            setTriggerValue
        } = component;

        const reviewBox =
            parentElement.querySelector("#reviewBox");

        const suggestionsBox =
            parentElement.querySelector("#suggestions");

        const analyzeButton =
            parentElement.querySelector("#analyzeButton");


        // -------------------------------------------------
        // DATA
        // -------------------------------------------------

        const sentences =
            (data && data.sentences) || [];

        const words =
            (data && data.words) || [];


        // -------------------------------------------------
        // CURRENT SUGGESTIONS
        // -------------------------------------------------

        let currentSuggestions = [];

        let activeIndex = -1;


        // -------------------------------------------------
        // SET REVIEW STATE
        // -------------------------------------------------

        function updateReview() {

            setStateValue(
                "review",
                reviewBox.value
            );
        }


        // -------------------------------------------------
        // HIDE SUGGESTIONS
        // -------------------------------------------------

        function hideSuggestions() {

            suggestionsBox.innerHTML = "";

            suggestionsBox.style.display = "none";

            currentSuggestions = [];

            activeIndex = -1;
        }


        // -------------------------------------------------
        // SHOW SUGGESTIONS
        // -------------------------------------------------

        function showSuggestions(items) {

            suggestionsBox.innerHTML = "";

            currentSuggestions = items;

            activeIndex = -1;


            if (!items.length) {

                hideSuggestions();

                return;
            }


            items.forEach(
                function(item, index) {

                    const div =
                        document.createElement("div");

                    div.className =
                        "suggestion-item";

                    div.textContent = item;


                    div.addEventListener(
                        "mousedown",
                        function(event) {

                            event.preventDefault();

                            selectSuggestion(
                                item
                            );
                        }
                    );


                    suggestionsBox.appendChild(
                        div
                    );
                }
            );


            suggestionsBox.style.display =
                "block";
        }


        // -------------------------------------------------
        // FIND SUGGESTIONS
        // -------------------------------------------------

        function findSuggestions() {

            const text =
                reviewBox.value;

            const trimmed =
                text.trim();

            if (!trimmed) {

                hideSuggestions();

                return;
            }


            const lower =
                trimmed.toLowerCase();


            // ---------------------------------------------
            // SENTENCE SUGGESTIONS
            // ---------------------------------------------

            let matches =
                sentences.filter(
                    function(sentence) {

                        return sentence
                            .toLowerCase()
                            .startsWith(lower);
                    }
                );


            // ---------------------------------------------
            // CURRENT UNFINISHED PART
            // ---------------------------------------------

            if (!matches.length) {

                const parts =
                    text.split(
                        /[.!?]\\s*/
                    );

                const currentPart =
                    parts[parts.length - 1]
                        .trim()
                        .toLowerCase();


                if (currentPart) {

                    matches =
                        sentences.filter(
                            function(sentence) {

                                return sentence
                                    .toLowerCase()
                                    .startsWith(
                                        currentPart
                                    );
                            }
                        );
                }
            }


            // ---------------------------------------------
            // WORD FALLBACK
            // ---------------------------------------------

            if (!matches.length) {

                const wordsInReview =
                    text.split(/\s+/);

                const lastWord =
                    wordsInReview[
                        wordsInReview.length - 1
                    ]
                    .toLowerCase()
                    .replace(
                        /[^a-z']/g,
                        ""
                    );


                if (lastWord) {

                    matches =
                        words.filter(
                            function(word) {

                                return word
                                    .toLowerCase()
                                    .startsWith(
                                        lastWord
                                    );
                            }
                        );
                }
            }


            // Maximum 8 suggestions
            matches =
                matches.slice(0, 8);


            showSuggestions(
                matches
            );
        }


        // -------------------------------------------------
        // SELECT SUGGESTION
        // -------------------------------------------------

        function selectSuggestion(
            suggestion
        ) {

            const text =
                reviewBox.value;


            // ---------------------------------------------
            // EMPTY TEXT
            // ---------------------------------------------

            if (!text.trim()) {

                reviewBox.value =
                    suggestion;

            }

            else {

                // -----------------------------------------
                // CHECK FOR COMPLETED SENTENCES
                // -----------------------------------------

                const match =
                    text.match(
                        /^(.*[.!?]\\s+)([^.!?]*)$/
                    );


                if (match) {

                    const previousText =
                        match[1];

                    const currentText =
                        match[2];


                    // If suggestion matches
                    // current unfinished part
                    if (
                        suggestion
                            .toLowerCase()
                            .startsWith(
                                currentText
                                    .trim()
                                    .toLowerCase()
                            )
                    ) {

                        reviewBox.value =
                            previousText +
                            suggestion;
                    }

                    else {

                        reviewBox.value =
                            previousText +
                            suggestion;
                    }

                }

                else {

                    // -------------------------------------
                    // WORD SUGGESTION
                    // -------------------------------------

                    const wordsInText =
                        text.split(/\s+/);

                    const last =
                        wordsInText[
                            wordsInText.length - 1
                        ];


                    const isWordMatch =
                        word_suggestions_check(
                            suggestion,
                            last
                        );


                    if (isWordMatch) {

                        wordsInText[
                            wordsInText.length - 1
                        ] = suggestion;

                        reviewBox.value =
                            wordsInText.join(" ");

                    }

                    else {

                        reviewBox.value =
                            suggestion;
                    }
                }
            }


            // Put cursor at end
            reviewBox.focus();

            reviewBox.selectionStart =
                reviewBox.value.length;

            reviewBox.selectionEnd =
                reviewBox.value.length;


            updateReview();

            hideSuggestions();
        }


        // -------------------------------------------------
        // WORD MATCH CHECK
        // -------------------------------------------------

        function word_suggestions_check(
            suggestion,
            lastWord
        ) {

            const cleanLast =
                lastWord
                    .toLowerCase()
                    .replace(
                        /[^a-z']/g,
                        ""
                    );

            return suggestion
                .toLowerCase()
                .startsWith(
                    cleanLast
                );
        }


        // -------------------------------------------------
        // KEYBOARD NAVIGATION
        // -------------------------------------------------

        reviewBox.addEventListener(
            "keydown",
            function(event) {

                const items =
                    suggestionsBox.querySelectorAll(
                        ".suggestion-item"
                    );


                if (
                    event.key === "ArrowDown"
                    && items.length
                ) {

                    event.preventDefault();

                    activeIndex =
                        (activeIndex + 1)
                        % items.length;

                    updateActiveItem(
                        items
                    );
                }


                else if (
                    event.key === "ArrowUp"
                    && items.length
                ) {

                    event.preventDefault();

                    activeIndex =
                        (activeIndex - 1 +
                        items.length)
                        % items.length;

                    updateActiveItem(
                        items
                    );
                }


                else if (
                    event.key === "Enter"
                    && activeIndex >= 0
                    && items.length
                ) {

                    event.preventDefault();

                    selectSuggestion(
                        currentSuggestions[
                            activeIndex
                        ]
                    );
                }


                else if (
                    event.key === "Escape"
                ) {

                    hideSuggestions();
                }
            }
        );


        // -------------------------------------------------
        // ACTIVE SUGGESTION
        // -------------------------------------------------

        function updateActiveItem(
            items
        ) {

            items.forEach(
                function(item, index) {

                    if (
                        index === activeIndex
                    ) {

                        item.classList.add(
                            "active"
                        );

                    }

                    else {

                        item.classList.remove(
                            "active"
                        );
                    }
                }
            );
        }


        // -------------------------------------------------
        // TYPING EVENT
        // -------------------------------------------------

        reviewBox.addEventListener(
            "input",
            function() {

                updateReview();

                findSuggestions();
            }
        );


        // -------------------------------------------------
        // ANALYZE BUTTON
        // -------------------------------------------------

        analyzeButton.addEventListener(
            "click",
            function() {

                const review =
                    reviewBox.value.trim();


                if (!review) {

                    return;
                }


                hideSuggestions();

                updateReview();


                // Trigger analysis only
                // when the button is clicked
                setTriggerValue(
                    "analyze",
                    review
                );
            }
        );


        // -------------------------------------------------
        // INITIAL VALUE
        // -------------------------------------------------

        if (
            data &&
            data.review &&
            !reviewBox.value
        ) {

            reviewBox.value =
                data.review;
        }


        // -------------------------------------------------
        // INITIAL STATE
        // -------------------------------------------------

        updateReview();
    }
    """,

    isolate_styles=True
)


# =========================================================
# CALLBACK FUNCTIONS
# =========================================================

def on_review_change():

    pass


def on_analyze_change():

    pass


# =========================================================
# COMPONENT
# =========================================================

result = review_autocomplete(

    data={
        "review": st.session_state.get(
            "review_text",
            ""
        ),

        "sentences": sentence_suggestions,

        "words": word_suggestions
    },

    default={
        "review": st.session_state.get(
            "review_text",
            ""
        )
    },

    key="review_autocomplete",

    on_review_change=on_review_change,

    on_analyze_change=on_analyze_change
)


# =========================================================
# STORE REVIEW
# =========================================================

if result is not None:

    if hasattr(
        result,
        "review"
    ):

        if result.review is not None:

            st.session_state[
                "review_text"
            ] = result.review


# =========================================================
# TITLE
# =========================================================

st.title(
    "🎬 Movie Review Sentiment Analysis"
)

st.write(
    "Enter a movie review and the machine learning "
    "model will predict whether the sentiment is "
    "positive or negative."
)


# =========================================================
# ANALYZE RESULT
# =========================================================

analysis_request = None


if result is not None:

    if hasattr(
        result,
        "analyze"
    ):

        analysis_request = result.analyze


# =========================================================
# ANALYZE REVIEW
# =========================================================

if analysis_request:

    review = str(
        analysis_request
    ).strip()


    # -----------------------------------------------------
    # EMPTY REVIEW
    # -----------------------------------------------------

    if not review:

        st.warning(
            "⚠️ Please enter a review."
        )

        st.stop()


    # -----------------------------------------------------
    # DISPLAY ENTERED REVIEW
    # -----------------------------------------------------

    st.subheader(
        "📝 Your Review"
    )

    st.write(
        review
    )


    # -----------------------------------------------------
    # REVIEW STATISTICS
    # -----------------------------------------------------

    word_count = len(
        review.split()
    )

    character_count = len(
        review
    )


    st.subheader(
        "📊 Review Statistics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Words",
            word_count
        )

    with col2:

        st.metric(
            "Characters",
            character_count
        )


    # -----------------------------------------------------
    # CLEAN REVIEW
    # -----------------------------------------------------

    cleaned_review = clean_text(
        review
    )


    if not cleaned_review:

        st.warning(
            "⚠️ Unable to analyze this review."
        )

        st.stop()


    # =====================================================
    # AMBIGUOUS REVIEW DETECTION
    # =====================================================

    ambiguous_reviews = [

        "it was okay",
        "it was ok",
        "it was average",
        "nothing special",
        "could be better",
        "not bad",
        "not too bad",
        "just okay",
        "just ok",
        "it was fine",
        "pretty average",
        "so so",
        "mixed feelings",
        "nothing great",
        "nothing amazing"
    ]


    normalized_review = cleaned_review.lower().strip()


    is_ambiguous = any(
        phrase in normalized_review
        for phrase in ambiguous_reviews
    )


    if is_ambiguous:

        st.warning(
            "⚠️ AMBIGUOUS REVIEW"
        )

        st.info(
            "This review expresses a neutral or mixed "
            "opinion. The trained model only predicts "
            "positive or negative sentiment."
        )

        st.stop()


    # =====================================================
    # TF-IDF TRANSFORMATION
    # =====================================================

    review_vector = tfidf.transform(
        [cleaned_review]
    )


    # =====================================================
    # UNKNOWN WORD CHECK
    # =====================================================

    if review_vector.nnz == 0:

        st.warning(
            "⚠️ Unable to analyze this review because "
            "none of its words are recognized by the "
            "trained model."
        )

        st.stop()


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(
        review_vector
    )[0]


    # =====================================================
    # PREDICTION PROBABILITY
    # =====================================================

    probabilities = model.predict_proba(
        review_vector
    )[0]


    confidence = (
        max(probabilities) * 100
    )


    positive_probability = (
        probabilities[1] * 100
    )

    negative_probability = (
        probabilities[0] * 100
    )


    # =====================================================
    # DISPLAY PREDICTION
    # =====================================================

    st.subheader(
        "🎯 Prediction"
    )


    if confidence < 60:

        st.warning(
            "⚠️ UNCERTAIN"
        )

        st.write(
            f"Model confidence: "
            f"**{confidence:.2f}%**"
        )


    elif prediction == 1:

        st.success(
            "😊 POSITIVE"
        )

        st.write(
            f"Confidence: "
            f"**{confidence:.2f}%**"
        )


    else:

        st.error(
            "😞 NEGATIVE"
        )

        st.write(
            f"Confidence: "
            f"**{confidence:.2f}%**"
        )


    # =====================================================
    # CONFIDENCE BAR
    # =====================================================

    st.progress(
        min(int(confidence), 100)
    )


    # =====================================================
    # SENTIMENT PROBABILITY
    # =====================================================

    st.subheader(
        "📈 Sentiment Probability"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"😊 Positive: "
            f"**{positive_probability:.2f}%**"
        )


    with col2:

        st.write(
            f"😞 Negative: "
            f"**{negative_probability:.2f}%**"
        )


    # =====================================================
    # CONFIDENCE MESSAGE
    # =====================================================

    if confidence >= 80:

        st.info(
            "✓ The model has high confidence "
            "in this prediction."
        )

    elif confidence >= 60:

        st.info(
            "✓ The model has moderate confidence "
            "in this prediction."
        )

    else:

        st.warning(
            "⚠️ The model has low confidence "
            "in this prediction."
        )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.subheader(
        "🤖 Model Information"
    )

    st.write(
        "**Algorithm:** Logistic Regression"
    )

    st.write(
        "**Feature Extraction:** TF-IDF"
    )

    st.write(
        "**Training Dataset:** IMDb 50,000 Reviews"
    )

    st.write(
        "**Test Accuracy:** Approximately 91.44%"
    )


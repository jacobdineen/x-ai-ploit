<template>
    <div class="faqs">
      <h1>FAQs</h1>
      <div class="faq-container" v-for="(item, index) in faqItems" :key="index">
        <div class="faq-question" @click="toggleAnswer(index)">
          {{ item.question }}
        </div>
        <div class="faq-answer" v-if="activeIndex === index">
          {{ item.answer }}
        </div>
      </div>
    </div>
  </template>

  <script>
  export default {
    data() {
      return {
        faqItems: [
                {
                    question: 'What is X(AI)PLOIT?',
                    answer: 'X(AI)PLOIT is an advanced framework for explaining AI-driven exploit predictions...'
                },
                {
                    question: 'How do I use it?',
                    answer: 'To use X(AI)PLOIT, navigate to the "Explain" section and input your parameters...'
                },
                {
                    question: 'What technologies does it use?',
                    answer: 'X(AI)PLOIT is built using a combination of machine learning algorithms and web technologies including Vue.js...'
                },
                {
                    question: 'Is it open source?',
                    answer: 'Yes, the codebase is hosted on GitHub and open for contributions...'
                },
                {
                    question: 'How can I contribute?',
                    answer: 'To contribute, please fork the repository, make your changes, and submit a pull request...'
                },
                {
                    question: 'What are the system requirements?',
                    answer: 'X(AI)PLOIT requires a modern web browser and works best on systems with at least 4GB of RAM...'
                },
                {
                    question: 'How do I report issues?',
                    answer: 'Please report any issues you encounter via our GitHub issues page...'
                }
                ],
        activeIndex: null,
      };
    },
    methods: {
      toggleAnswer(index) {
        this.activeIndex = this.activeIndex === index ? null : index;
      }
    }
  };
  </script>

  <style scoped>
  .faqs {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  margin: 20px;
  font-family: 'Nunito', sans-serif; /* You may change the font family */
}

.faq-container {
  width: 80%;
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0px 2px 12px rgba(0, 0, 0, 0.1);
}

.faq-question {
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  background: linear-gradient(90deg, rgba(55,27,100,1) 0%, rgba(65,155,198,1) 50%, rgba(55,27,100,1) 100%);
  color: #fff;
  padding: 15px;
  cursor: pointer;
  font-weight: 600; /* Making the text bold */
}

.faq-answer {
  padding: 15px;
  background-color: #f1f1f1;
  border-bottom-left-radius: 8px;
  border-bottom-right-radius: 8px;
  font-size: 0.9em; /* Adjust font size */
  line-height: 1.6; /* Adjust line height */
}
  </style>

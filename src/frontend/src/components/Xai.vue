<template>
    <div id="app" class="center-content">
      <div class="container">
        <div class="message-box" v-if="filteredPapers.length === 0">
          <div class="message">{{ message }}</div>
        </div>
        <div class="search-box">
          <input type="text" v-model="searchTerm" @input="filterCandidates" placeholder="Search by Paper ID" />
          <ul v-if="candidates.length > 0" class="suggestions">
            <li v-for="candidate in candidates" :key="candidate" @click="selectCandidate(candidate)">
              {{ candidate }}
            </li>
          </ul>
        </div>
        <vue-good-table
          v-if="filteredPapers.length > 0"
          :columns="columns"
          :rows="filteredPapers"
          :pagination-options="{ enabled: true, mode: 'pages' }"
          :sort-options="{ enabled: true }"
        >
        </vue-good-table>
      </div>
    </div>
  </template>
  
  
  <script>
import axios from 'axios';
import { VueGoodTable } from 'vue-good-table';

export default {
  components: {
    VueGoodTable,
  },
  data() {
    return {
      message: 'Fetching data...',
      papers: [],
      searchTerm: '',
      candidates: [],
      columns: [
        { label: 'Paper ID', field: 'paper_id', sortable: true },
        { label: 'Title', field: 'title', sortable: true },
        { label: 'Abstract', field: 'abstract', sortable: false },
      ],
    };
  },
  computed: {
    filteredPapers() {
      return this.papers.filter(paper => paper.paper_id.toLowerCase().includes(this.searchTerm.toLowerCase()));
    },
  },
  methods: {
    filterCandidates() {
      this.candidates = this.papers
        .map(paper => paper.paper_id)
        .filter(id => id.toLowerCase().includes(this.searchTerm.toLowerCase()))
        .slice(0, 5); // Limit number of suggestions
    },
    selectCandidate(candidate) {
      this.searchTerm = candidate;
      this.candidates = [];
    },
  },
  created() {
    axios.get('http://localhost:3001/api/message', {
      headers: {
        'Content-Type': 'application/json',
      },
    })
    .then(response => {
      this.papers = response.data;
      if (this.papers.length === 0) {
        this.message = 'No papers found.';
      }
    })
    .catch(error => {
      console.error('An error occurred while fetching data:', error);
      this.message = 'An error occurred while fetching data.';
    });
  },
};
</script>

  

<style>
.center-content {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}
.search-box {
  position: relative;
}
.suggestions {
  position: absolute;
  width: 100%;
  border: 1px solid #ccc;
  background: #fff;
  z-index: 1;
  list-style-type: none;
  margin: 0;
  padding: 0;
}
</style>

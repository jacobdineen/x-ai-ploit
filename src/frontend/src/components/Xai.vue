<template>
  <div id="app" class="center-content">
      <div class="container">
          <!-- CVE ID input and explanation button -->
          <div class="input-box">
            <input list="cve_ids" type="text" v-model="cveIdInput" placeholder="Enter CVE ID" />
            <datalist id="cve_ids">
              <option v-for="cve_id in suggestions.cve_ids" :value="cve_id" :key="cve_id"></option>
          </datalist>


            <input list="hashes" type="text" v-model="hashInput" placeholder="Enter Hash" />
            <datalist id="hashes">
                <option v-for="hash in suggestions.hashes" :value="hash" :key="hash"></option>
            </datalist>
            <button @click="fetchExplanation">Generate</button>
          </div>
          
          <div v-if="isLoading" class="progress-container">
            <div class="progress-bar" :style="{width: progress + '%'}"></div>
            <span class="progress-message">Generating explanations, hang tight...</span>
        </div>

          <!-- SHAP Values Display Container -->
          <iframe id="shapPlotFrame" style="width:100%; height:400px; border:1px solid red;"></iframe>
          <!-- SHAP Values Image Display (optional, if you still want to display the image) -->
          <div>
              <img :src="'data:image/png;base64,' + shapImage" alt="SHAP Visualization" />
          </div>
      </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
        cveIdInput: 'cve-2013-6234',  // Default value
        hashInput: '3cbdafc2a8811dbc3e6ee66bdc29ef72719e0a1e298017f3852bbdc54219b41e',  // Default value
        suggestions: {
            cve_ids: [],
            hashes: []
        },
        isLoading: false,
        progress: 0
    };
},

created() {
    this.fetchSuggestions();
},

watch: {
    cveIdInput(newCveId) {
      this.fetchHashesForCVE(newCveId);
    }
  },


  methods: {
    startProgressBar() {
        this.progress = 0;
        const interval = setInterval(() => {
            this.progress += 1;
            if (this.progress >= 100) {
                clearInterval(interval);
                this.isLoading = false;
            }
        }, 600); // This will increment the progress by 1% every 600ms for 60 seconds
    },

    fetchHashesForCVE(cve_id) {
      axios.post('http://localhost:3001/api/get_hashes_for_cve', { cve_id: cve_id })
        .then(response => {
            this.suggestions.hashes = response.data.hashes;
        })
        .catch(error => {
            console.error("Error fetching hashes for CVE:", error);
        });
    },

    fetchSuggestions() {
        axios.get('http://localhost:3001/api/get_suggestions')
            .then(response => {
                this.suggestions = response.data;
            })
            .catch(error => {
                console.error("Error fetching suggestions:", error);
            });
    },

    fetchExplanation() {
    this.startProgressBar(); // Start the progress bar
    this.isLoading = true; // Set isLoading to true to show the progress bar

    axios.post('http://localhost:3001/api/explain', { 
        cve_id: this.cveIdInput, 
        hash: this.hashInput
    })
    .then(response => {
        const iframe = document.getElementById('shapPlotFrame');
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        doc.open();
        doc.write(response.data.shap_plot);
        doc.close();

        this.isLoading = false; // Set isLoading back to false once done
    })
    .catch(error => {
        console.error("Error fetching SHAP values:", error);
        this.isLoading = false; // Set isLoading to false even on error
    });
}

  }
}
</script>

<style>
.progress-container {
    position: relative;
    width: 100%;
    background-color: #f3f3f3;
    margin-top: 15px;
    height: 20px;
    border-radius: 3px;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    background-color: #4CAF50;
    transition: width 0.6s ease;
}

.center-content {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}
.input-box {
  margin-bottom: 20px;
}
</style>

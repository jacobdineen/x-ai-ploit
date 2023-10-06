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

        <div class="plots-container">
            <!-- Existing SHAP Text Plot Display -->
            <iframe id="shapPlotFrameText" class="plot-frame" style="border:1px solid red;"></iframe>

            <!-- New SHAP Bar Plot Display -->
            <img :src="shapPlotBarImage" class="plot-frame" style="border:1px solid blue;" />
        </div>

        <div v-if="exploitNotLikelyScore">
          P(exploitability = 0) = {{ exploitNotLikelyScore }}
      </div>
      <div v-if="exploitLikelyScore">
          P(exploitability = 1) = {{ exploitLikelyScore }}
      </div>

 
      </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
        cveIdInput: 'cve-2015-8103',  // Default value
        hashInput: '5b4fbac60182b69e9a0417ea726276c87c101335d6fbdc5e8c9a9a429235c655',  // Default value
        suggestions: {
            cve_ids: [],
            hashes: []
        },
        isLoading: false,
        progress: 0,
        shapPlotBarImage: null, // Add this line
        exploitNotLikelyScore: 0,
        exploitLikelyScore: 0,
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
        }, 900); // This will increment the progress by 1% every 600ms for 60 seconds
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
      this.startProgressBar();
      this.isLoading = true;

      axios.post('http://localhost:3001/api/explain', { 
          cve_id: this.cveIdInput, 
          hash: this.hashInput
      })
      .then(response => {
          const iframeText = document.getElementById('shapPlotFrameText');
          const docText = iframeText.contentDocument || iframeText.contentWindow.document;
          docText.open();
          docText.write(response.data.shap_plot_text);
          docText.close();

          this.shapPlotBarImage = response.data.shap_plot_bar;

          // Check if "preds" exists, is an array, and contains at least one item
          const preds = response.data.preds;
          if (preds && Array.isArray(preds) && preds.length > 0 && preds[0]) {
              preds[0].forEach(pred => {
                  if (pred.label === 'exploit_not_likely') {
                      this.exploitNotLikelyScore = pred.score;
                  } else if (pred.label === 'exploit_likely') {
                      this.exploitLikelyScore = pred.score;
                  }
              });
          }
      })
      .catch(error => {
          console.error("Error fetching SHAP values:", error);
      });
  }
}
}
</script>

<style>
/* Styles for the plots container */
.plots-container {
    display: flex;
    justify-content: flex-start; /* Ensures horizontal centering */
    align-items: left; /* Ensures vertical centering */
    flex-wrap: nowrap;
    width: 160%; /* Change this from 150% to 100% */
}

.plot-frame {
    flex: 1; 
    width: calc(50% - 10px); /* Adjusting for a 10px gap between plots. 50% since there are 2 plots */
    height: 400px;
    margin: 0 5px;
}

/* If the input boxes are getting obscured, ensure their container has a higher z-index */
.input-box {
    margin-bottom: 20px;
    z-index: 10;
    position: relative;
}



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

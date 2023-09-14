<template>
    <div class="container">
      <div v-if="showIntro">
        <section id="introduction" class="fall">
          <h1>
            <span :class="{'color-emph': emph[0]}">X</span>(<span :class="{'color-emph': emph[1]}">A</span><span :class="{'color-emph': emph[2]}">I</span>)<span :class="{'color-emph': emph[3]}">P</span><span :class="{'color-emph': emph[4]}">L</span><span :class="{'color-emph': emph[5]}">O</span><span :class="{'color-emph': emph[6]}">I</span><span :class="{'color-emph': emph[7]}">T</span>
            : e<span :class="{'color-emph': emph[0]}">X</span>plainable 
            <span :class="{'color-emph': emph[1]}">A</span><span :class="{'color-emph': emph[2]}">I</span> for 
            ex<span :class="{'color-emph': emph[3]}">P</span><span :class="{'color-emph': emph[4]}">L</span><span :class="{'color-emph': emph[5]}">O</span><span :class="{'color-emph': emph[6]}">I</span><span :class="{'color-emph': emph[7]}">T</span> Prediction
          </h1>
        </section>
      </div>
      <div v-if="showSefcom">
        <section id="sefcom" class="fall">
          <img src="@/assets/SEFCOM.png" alt="SEFCOM" class="sefcom-logo" />
        </section>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        showIntro: false,
        showSefcom: false,
        emph: new Array(8).fill(false)
      };
    },
    mounted() {
      setTimeout(() => {
        this.showIntro = true;
        let delay = 2000;  // Increased from 1000 to 2000 for slower start
        for (let i = 0; i < this.emph.length; i++) {
          setTimeout(() => {
            this.$set(this.emph, i, true);
          }, delay);
          delay += 400;  // Increased from 200 to 400 for slower emphasis
        }
      }, 400);  // You can adjust this as well
  
      setTimeout(() => {
        this.showSefcom = true;
      }, 800);  // You can adjust this as well
    }
  }
  </script>

  <style scoped>
  .container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh; /* 100% of the viewport height */
  width: 100vw; /* 100% of the viewport width */
}
  .fall {
    animation: fallAnimation 1s ease forwards;
    transform: translateY(-200vh);
    opacity: 0;
  }
  
    .fade-enter-active, .fade-leave-active {
    transition: opacity 1s;
  }
  .fade-enter, .fade-leave-to {
    opacity: 0;
  }

  
  .sefcom-logo {
    width: 200px;
    height: auto;
  }
  
  .color-emph {
    color: #007bff; /* Pick any color you prefer */
  }

  .emph {
    font-size: 1.5em;
    font-weight: bold;
  }

  .less-emph {
    font-size: 1.2em;
    font-weight: semi-bold;
  }
  
  @keyframes fallAnimation {
  to {
    transform: translateY(0);
    opacity: 1;
  }
}
  </style>
  
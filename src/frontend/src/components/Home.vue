<template>
    <div id="app">
      <div class="header">
        <div class="container">
          <nav>
            <router-link to="/">Home</router-link>
            <router-link to="/another">Another Page</router-link>
          </nav>
        </div>
      </div>
      <router-view></router-view>
    </div>
  </template>
  
  
  <script>
  import axios from 'axios';
  
  export default {
    data() {
      return {
        message: 'Fetching data...'
      };
    },
    created() {
      axios.get('http://localhost:3001/api/message', {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      .then(response => {
        this.message = response.data.message;
      })
      .catch(error => {
        console.error('An error occurred while fetching data.');
        console.error('Complete Error Object: ', error);
        this.message = 'An error occurred while fetching data.';
      });
    }
  }
  </script>
  
  <style>
  /* General Layout */
  #app {
    font-family: 'Nunito', sans-serif;
    background-color: #fafafa;
    height: 100vh;
  }
  
  .container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
  }
  
  /* Title Styles */
  .title {
    font-size: 3.5rem;
    color: #3498db;
  }
  
  /* Animated Title */
  .animated-title {
    animation: popIn 1s ease forwards;
  }
  
  /* Message Box Styles */
  .message-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  }
  
  /* Message Styles */
  .message {
    font-size: 1.5rem;
    color: #333;
    text-align: center;
  }
  
  /* Animated Message */
  .animated-message {
    animation: fadeIn 2s ease forwards;
  }
  
  /* Animations */
  @keyframes popIn {
    0% { transform: scale(0); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
  }
  
  @keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
  }
  </style>
  
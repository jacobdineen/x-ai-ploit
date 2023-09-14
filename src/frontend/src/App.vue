<template>
    <div id="app">
      <b-navbar toggleable="lg" type="dark" variant="danger">
        <b-navbar-brand href="#">
          <router-link to="/">Seer</router-link>
        </b-navbar-brand>
        <b-navbar-toggle target="nav-collapse"></b-navbar-toggle>
        <b-collapse id="nav-collapse" is-nav>
          <b-navbar-nav>
            <b-nav-item href="#">
              <router-link to="/intention">Intention Label</router-link>
            </b-nav-item>
            <b-nav-item href="#">
              <router-link to="/interactive">Interactive Module</router-link>
            </b-nav-item>
          </b-navbar-nav>
        </b-collapse>
      </b-navbar>
      <router-view></router-view>
    </div>
  </template>
  
  <script>
  
  export default {
    name: 'App',
    components: {
    }
  }
  </script>
  
  <style>
  #app {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
  }
  
  nav a {
    color: #e2bdbd;
  }
  
  nav a:hover {
    color: white;
    text-decoration: none;
  }
  </style>
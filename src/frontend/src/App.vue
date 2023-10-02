<template>
  <div id="app">
    <b-navbar id="custom-navbar" fixed="top">
    <b-navbar-brand>
        <router-link to="/" class="brand-link">X(AI)PLOIT</router-link>
      </b-navbar-brand>
      <b-navbar-toggle target="nav-collapse"></b-navbar-toggle>
      <b-collapse id="nav-collapse" is-nav>
        <b-navbar-nav>
          <b-nav-item>
            <router-link to="/predictions" class="nav-link predictions">Predict</router-link>
          </b-nav-item>
          <b-nav-item>
            <router-link to="/explain" class="nav-link explain">Explain</router-link>
          </b-nav-item>
          <b-nav-item>
          <router-link to="/faqs" class="nav-link">FAQs</router-link>
        </b-nav-item>
        </b-navbar-nav>
        
        <b-navbar-nav class="ml-auto">  <!-- Push content to the far right -->
          <b-nav-item>
            <img src="@/assets/SEFCOM.png" alt="SEFCOM" class="sefcom-logo"/>
          </b-nav-item>
        </b-navbar-nav>
      </b-collapse>
    </b-navbar>
    <router-view></router-view>
    <footer class="site-footer">
  <div class="footer-content">
    <div>An initiative by the Laboratory of Security Engineering for Future Computing (SEFCOM).</div>
    <div>For inquiries contact: <a href="mailto:jdineen@asu.edu">jdineen@asu.edu</a></div>
  </div>
</footer>

  </div>
</template>


<script>
export default {
  
  name: 'App',
}
</script>

<style>
router-view {
  margin-top: 56px; /* For the navbar */
  padding-bottom: 50px; /* For the footer, adjust as needed */
}

.site-footer {
  background-color: #2c3e50; /* dark blue-grey */
  color: #ecf0f1; /* off-white */
  padding: 10px 0; /* top-bottom padding */
  font-size: 14px; /* smaller text */
}

.footer-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: auto;
  padding: 0 20px; /* left-right padding */
}

.site-footer a {
  color: #3498db; /* brighter blue */
  text-decoration: none;
}

.site-footer a:hover {
  text-decoration: underline;
}


.sefcom-logo {
    width: 100px;  /* You can adjust the dimensions as needed */
    height: auto;
  }

#app {
  font-family: 'Nunito', sans-serif;
}

#custom-navbar {
  background: linear-gradient(90deg, rgba(85,37,130,1) 0%, rgba(95,195,228,1) 50%, rgba(85,37,130,1) 100%);
}


.brand-link {
  color: white;
  font-size: 1.5rem;
  transition: all 0.3s ease;
}

.brand-link:hover {
  color: lightgreen;
  text-shadow: 0px 0px 5px #fff;
  transform: scale(1.1);
}

.nav-link {
  color: white;  /* Making it visible even without hover */
  transition: all 0.3s ease;
}

.nav-link.explain { /* Specific class for 'Explain' */
  color: #FFFFFF;  /* Changing it to bright white */
}

/* Comment out or remove hover for .nav-link.explain */
/*.nav-link.explain:hover {
  background: orange;
  color: white;
}*/

.nav-link:hover {
  color: yellow;
  text-shadow: 0px 0px 5px #fff;
}

</style>

import React, { useEffect, useState } from 'react';

function App() {
  const [message, setMessage] = useState('Loading...');

  useEffect(() => {
    // Update this URL to point to your backend service
    fetch('http://localhost:3001/api/message')
      .then(response => response.json())
      .then(data => setMessage(data.message))
      .catch(err => setMessage(`Error: ${err}`));
  }, []);  // Empty dependency array means this useEffect runs once when the component mounts

  return (
    <div className="App">
      <h1>{message}</h1>
    </div>
  );
}

export default App;

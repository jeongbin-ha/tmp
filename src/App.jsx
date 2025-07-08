
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import TicketingPage from './pages/ticketingpage/TicketingPage.jsx';
import BoardPage from './pages/boardpage/BoardPage';

const App = () => {
  return (
    <Routes>
      <Route path="/ticketing/*" element={<TicketingPage />} />
      <Route path="/board/*" element={<BoardPage />} />
    </Routes>
  );
};

export default App;
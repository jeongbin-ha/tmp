
import React from 'react';
import { Header as HeaderContainer, BackButton, PageTitle } from '../styles/commonStyles';

const Header = ({ onBack }) => {
  return (
    <HeaderContainer>
      <BackButton onClick={onBack}>〈</BackButton>
      <PageTitle>예매</PageTitle>
    </HeaderContainer>
  );
};

export default Header;
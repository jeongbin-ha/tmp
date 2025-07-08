
import React, { useState, useEffect } from 'react';
import { AppContainer, MainContent, ScrollableContent } from './styles/commonStyles';
import Header from './components/Header';
import HeaderPC from './components/HeaderPC';
import Step1 from './pages/Step1';
import Step2 from './pages/Step2';
import Step3 from './pages/Step3';
import useTicketing from './hooks/useTicketing';
import useResponsive from './hooks/useResponsive';

const TicketingPage = () => {
  const ticketing = useTicketing();
  const { step, goToPreviousStep, getCurrentStepContent } = ticketing;
  const isPC = useResponsive();
  const currentContent = getCurrentStepContent();

  // 반응형 헤더 렌더링
  const renderHeader = () => {
    if (isPC && step === 5) return null;

    if (isPC) 
      return <HeaderPC currentStep={step} />
    else
      return <Header onBack={goToPreviousStep} />
  }

  // 현재 단계에 맞는 컴포넌트 렌더링
  const renderStep = () => {
    if (isPC) {
      switch (step) {   // PC
        case 1:
          return <Step1 ticketing={ticketing} />;
        case 2:
        case 3:
        case 4:
          return <Step2 ticketing={ticketing} />;
        case 5: 
          return <Step3 ticketing={ticketing} />;
        default:
          return <Step1 ticketing={ticketing} />;
      }
    } else {
      switch (step) {   // 모바일
        case 1:
          return <Step1 ticketing={ticketing} />;
        case 2:
          return <Step2 ticketing={ticketing} />;
        case 3:
          return <Step3 ticketing={ticketing} />;
        default:
          return <Step1 ticketing={ticketing} />;
      }
    }
  };

  return (
    <AppContainer>
      {renderHeader()}
      <ScrollableContent>
        {(currentContent !== 'discount' && currentContent !== 'delivery' && currentContent !== 'payment') && (
          <MainContent>
            {renderStep()}
          </MainContent>
        )}
        {(currentContent === 'discount' || currentContent === 'delivery' || currentContent === 'payment') && (
          <div style={{ margin: '0px 100px 0px 160px' }}>
            {renderStep()}
          </div>
        )}
      </ScrollableContent>
    </AppContainer>
  );
};

export default TicketingPage;
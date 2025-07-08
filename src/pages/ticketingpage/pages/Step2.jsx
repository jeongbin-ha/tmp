
import React from 'react';
import PosterInfo from '../components/PosterInfo';
import RadioGroup from '../components/RadioGroup';
import ActionButton from '../components/ActionButton';
import { 
  FormSection2, SelectionSection, SectionTitle, 
  EventTitle, EventVenue, EventPeriod, EventInfo, 
  EventLink, PcLayout, SummarySection, BackButton } from '../styles/commonStyles';
import { SummaryRow } from '../styles/summaryStyles';
import { 
  Divider, Label, AdditionalInputField,
  BankInfo, PaymentNotice, CheckboxContainer,
  CheckboxInput, CheckboxLabel, Showmore } from '../styles/formStyles'; 
import useResponsive from '../hooks/useResponsive';
import ShowMore from '../components/icons/ShowMore.svg';

const Step2 = ({ 
  ticketing: { 
    dateTime, 
    people, 
    eventInfo, 
    discountType, 
    paymentType,
    deliveryType, 
    nextActive,
    setDiscountType,
    setPaymentType, 
    setDeliveryType, 
    goToNextStep,
    goToPreviousStep,
    calculatePayment,
    studentId,
    setStudentId,
    depositorName,
    setDepositorName,
    termsAgreed,
    setTermsAgreed,
    getCurrentStepContent
  } 
}) => {

  const isPC = useResponsive();
  const currentContent = getCurrentStepContent();
  // PC: 개별 단계 / 모바일: 할인+수령+결제 통합
  const showDiscountOnly = isPC && currentContent === 'discount';
  const showDeliveryOnly = isPC && currentContent === 'delivery';
  const showPaymentOnly = isPC && currentContent === 'payment';
  const showAllOptions = !isPC && currentContent === 'options';

  const discountOptions = [
    { value: 'none', label: '없음' },
    { value: 'standard', label: '홍대생 할인' }
  ];

  const deliveryOptions = [
    { value: 'venue', label: '현장 수령' }
  ];

  const paymentOptions = [
    /* { value: 'bank', label: '입금 (주) 홍익극연구회' }, */   // 계좌이체 제거
    { value: 'pay', label: '카카오페이' }
  ];

  // 할인 선택 추가 컨텐츠 정의 > 모바일용
  const discountAdditionalContent = {
    ...(isPC ? {} : {
      standard: (
        <AdditionalInputField>
          <input
            type="text"
            placeholder="학번을 입력하세요"
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
          />
        </AdditionalInputField>
      )
    })
  };

  // 결제 수단 추가 컨텐츠 정의
  const paymentAdditionalContent = {
    /*  //계좌이체 제거
    bank: (
      <div style={{ marginTop: '-4px' }}>
        <BankInfo>
          국민은행 112233445566745
        </BankInfo>
        <AdditionalInputField style={{ marginTop: '8px' }}>
          <input
            type="text"
            placeholder="입금자명을 입력하세요"
            value={depositorName}
            onChange={(e) => setDepositorName(e.target.value)}
          />
        </AdditionalInputField>
        <PaymentNotice>
          예매후 24시간동안 미입금시 자동 취소됩니다.
        </PaymentNotice>
      </div>
    ),*/
    pay: (
      <CheckboxContainer style={{ marginLeft: isPC ? '0px' : '-22px' }}>
        {/* 카카오페이는 marginLeft: 0으로 조정 (이미 RadioGroup에서 22px 적용) */}
        <CheckboxInput
          type="checkbox"
          checked={termsAgreed}
          onChange={(e) => setTermsAgreed(e.target.checked)}
        />
        <CheckboxLabel>결제대행 서비스 이용 약관 동의</CheckboxLabel>
        <Showmore src={ShowMore} alt=">" /* onClick={ 추후 추가 } */ />
      </CheckboxContainer>
    )
  };

  const payment = calculatePayment();

  return (
    <>
      {!isPC && (
        <PosterInfo eventInfo={eventInfo} />
      )}
      {isPC && (
        <>
          <EventInfo style={{ marginTop: '0px' }}>
            <EventTitle>{eventInfo.title}</EventTitle>
            <EventLink src={ShowMore} alt=">" /*onClick={ 추후추가 }*/ />
          </EventInfo>
          <EventVenue>{eventInfo.venue}</EventVenue>
          <EventPeriod style={{ width: 'auto' }}>{eventInfo.period}</EventPeriod>
        </>
      )}
      
      <PcLayout>
        {(showDiscountOnly || showAllOptions) && (    // 할인 선택 - PC 2단계 & 모바일 2단계
          <FormSection2>
            {!isPC && ( <SectionTitle>할인 선택</SectionTitle> )}
            <SelectionSection>
                {!isPC && ( <Label>할인 선택</Label> )}
                {isPC && ( <label style={{fontSize: '20px', fontWeight: 'bold', color: '#000000'}}>할인 선택</label> )}
                <RadioGroup 
                options={discountOptions} 
                selectedValue={discountType} 
                onChange={setDiscountType} 
                additionalContent={discountAdditionalContent}
                />
                {isPC && discountType === 'standard' && (   // PC용 홍대생 할인 인풋
                  <>
                    <label style={{fontSize: '20px', fontWeight: 'bold', color: '#000000', marginTop: '32px'}}>할인 인증</label>
                    <SelectionSection>
                      <div style={{marginTop: '14px', display: 'flex', flexDirection: 'row'}}>
                        <div style={{width: '80px', fontSize: '16px', fontWeight: 'bold', color: '#000000'}}>이름</div>
                        <div style={{marginLeft: '40px', fontWeight: '500'}}>최윤경</div>
                      </div>
                      <div style={{marginTop: '14px', display: 'flex', flexDirection: 'row', alignItems: 'center'}}>
                        <div style={{width: '80px', fontSize: '16px', fontWeight: 'bold', color: '#000000'}}>학번 입력</div>
                        <AdditionalInputField>
                          <input
                            type="text"
                            placeholder="학번을 입력하세요"
                            value={studentId}
                            onChange={(e) => setStudentId(e.target.value)}
                          />
                        </AdditionalInputField>
                      </div>
                    </SelectionSection>
                  </>
                )}
            </SelectionSection>
          </FormSection2>
        )}
        
        {(showDeliveryOnly || (showAllOptions && discountType)) && (  // 수령 선택 - PC 3단계 & 모바일 2단계
          <FormSection2>
              {!isPC && ( <SectionTitle>티켓 수령</SectionTitle> )}
              <SelectionSection>
                  {!isPC && ( <Label>티켓 수령</Label> )}
                  {isPC && ( <label style={{fontSize: '20px', fontWeight: 'bold', color: '#000000'}}>티켓 수령</label> )}
                  <RadioGroup 
                  options={deliveryOptions} 
                  selectedValue={deliveryType} 
                  onChange={setDeliveryType} 
                  />
              </SelectionSection>
          </FormSection2>
        )}

        {(showPaymentOnly || (showAllOptions && discountType && deliveryType)) && (   // 결제 선택 - PC 4단계 & 모바일 2단계
          <FormSection2>
              {!isPC && ( <SectionTitle>결제</SectionTitle> )}
              <SelectionSection>
                  {!isPC && ( <Label>결재 수단</Label> )}
                  {isPC && ( <label style={{fontSize: '20px', fontWeight: 'bold', color: '#000000'}}>결제 수단</label> )}
                  <RadioGroup 
                  options={paymentOptions} 
                  selectedValue={paymentType} 
                  onChange={setPaymentType} 
                  additionalContent={paymentAdditionalContent}
                  />
              </SelectionSection>
          </FormSection2>
        )}

        {!isPC && ( <Divider /> )}
        
        <SummarySection>
          {!isPC && ( <SectionTitle>예매 내용</SectionTitle> )}
          <SummaryRow>
            <div>일시</div>
            <span>{dateTime || ''}</span>
          </SummaryRow>
          <SummaryRow>
            <div>인원</div>
            <span>{people || ''}</span>
          </SummaryRow>
          <SummaryRow>
            <div>티켓 금액</div>
            <span>{payment.basePrice.toLocaleString()}원</span>
          </SummaryRow>
          <SummaryRow>
            <div>할인</div>
            <span className={discountType ? "discount" : ""}>
              {discountType ? `-${payment.discountAmount.toLocaleString()}원` : ''}
            </span>
          </SummaryRow>
          <SummaryRow>
            <div>배송료</div>
            <span>{deliveryType ? `${payment.deliveryFee.toLocaleString()}원 (현장수령)` : ''}</span>
          </SummaryRow>
          <SummaryRow className="total">
            <div>총 결제 금액</div>
            {!isPC && ( <span>{(payment.totalPrice * people).toLocaleString()}원</span> )}
            {isPC && ( <span style={{fontSize: '24px', bottom: '4px'}}>{(payment.totalPrice * people).toLocaleString()}원</span> )}
          </SummaryRow>
        </SummarySection>
      </PcLayout>
      {!isPC && (
        <ActionButton isActive={nextActive} onClick={goToNextStep}>
          다음
        </ActionButton>
      )}
      {isPC && (
        <div style={{ marginLeft: '718px', marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '20px'}}>
          <ActionButton isActive={nextActive} onClick={goToNextStep}>
            다음
          </ActionButton>
          <BackButton onClick={goToPreviousStep}>이전</BackButton>
        </div>
      )}
    </>
  );
};

export default Step2;

import { useState, useEffect } from 'react';
import useResponsive from './useResponsive';

const useTicketing = () => {
  // 공연의 날짜+시간 MOCK data
  const dateTimeOptions = [
    { value: '2025.05.14 (수) 15:00', display: '2025.05.14 (수) 15:00' },
    { value: '2025.05.14 (수) 17:00', display: '2025.05.14 (수) 17:00' },
    { value: '2025.05.14 (수) 19:00', display: '2025.05.14 (수) 19:00' },
    { value: '2025.05.15 (목) 16:00', display: '2025.05.15 (목) 16:00' },
    { value: '2025.05.15 (목) 20:00', display: '2025.05.15 (목) 20:00' },
    { value: '2025.05.16 (금) 13:00', display: '2025.05.16 (금) 13:00' },
    { value: '2025.05.16 (금) 16:00', display: '2025.05.16 (금) 16:00' },
    { value: '2025.05.16 (금) 19:00', display: '2025.05.16 (금) 19:00' },
    { value: '2025.05.17 (토) 16:00', display: '2025.05.17 (토) 16:00' },
    { value: '2025.05.17 (토) 19:00', display: '2025.05.17 (토) 19:00' },
    { value: '2025.05.18 (일) 11:00', display: '2025.05.18 (일) 11:00' },
    { value: '2025.05.18 (일) 13:00', display: '2025.05.18 (일) 13:00' },
    { value: '2025.05.18 (일) 15:00', display: '2025.05.18 (일) 15:00' },
    { value: '2025.05.18 (일) 17:00', display: '2025.05.18 (일) 17:00' },
  ];
  const [step, setStep] = useState(1);
  const [dateTime, setDateTime] = useState(null);
  const [people, setPeople] = useState(null);
  const [discountType, setDiscountType] = useState(null);
  const [paymentType, setPaymentType] = useState(null);
  const [deliveryType, setDeliveryType] = useState(null);
  const [nextActive, setNextActive] = useState(false);
  const [studentId, setStudentId] = useState(''); // 학번 입력
  const [depositorName, setDepositorName] = useState(''); // 입금자명 입력
  const [termsAgreed, setTermsAgreed] = useState(false); // 약관 동의 체크
  const isPC = useResponsive(); //반응형

  // 최대 단계 수 계산
  const getMaxSteps = () => isPC ? 5 : 3; // PC: 1날짜,인원→2할인→3수령→4결제→5완료, 모바일: 1날짜→2옵션들→3완료
  // 현재 단계의 컨텐츠 타입 반환
  const getCurrentStepContent = () => {
    if (!isPC) {
      switch(step) {
        case 1: return 'datetime';  // 날짜 인원
        case 2: return 'options';   // 할인+수령+결제 통합
        case 3: return 'complete';
        default: return 'datetime';
      }
    } else {
      switch(step) {
        case 1: return 'datetime';  // 날짜 인원
        case 2: return 'discount';  // 할인
        case 3: return 'delivery';  // 수령
        case 4: return 'payment';   // 결제
        case 5: return 'complete';
        default: return 'datetime';
      }
    }
  };

  // 다음 버튼 활성화 여부 확인
  useEffect(() => {
    const currentContent = getCurrentStepContent();
    
    switch(currentContent) {
      case 'datetime':
        setNextActive(dateTime && people);
        break;
        
      case 'discount':  //pc 할인
        let discountValid = discountType !== null;
        if (discountType === 'standard') {
          discountValid = discountValid && studentId.trim() !== '';
        }
        setNextActive(discountValid);
        break;
        
      case 'delivery':  //pc 수령
        setNextActive(deliveryType !== null);
        break;
        
      case 'payment':   //pc 결제
        let paymentValid = paymentType !== null;
        if (paymentType === 'bank') {
          paymentValid = paymentValid && depositorName.trim() !== '';
        }
        if (paymentType === 'pay') {
          paymentValid = paymentValid && termsAgreed;
        }
        setNextActive(paymentValid);
        break;
        
      case 'options': // 모바일 통합 단계
        let optionsValid = discountType !== null && paymentType !== null && deliveryType !== null;
        if (discountType === 'standard') {
          optionsValid = optionsValid && studentId.trim() !== '';
        }
        if (paymentType === 'bank') {
          optionsValid = optionsValid && depositorName.trim() !== '';
        }
        if (paymentType === 'pay') {
          optionsValid = optionsValid && termsAgreed;
        }
        setNextActive(optionsValid);
        break;
        
      default:
        setNextActive(false);
    }
  }, [step, dateTime, people, discountType, paymentType, deliveryType, studentId, depositorName, termsAgreed, isPC]);

  // 다음 단계로 이동
  const goToNextStep = () => {
    if (nextActive) {
      setStep(step + 1);
    }
  };

  // 이전 단계로 이동
  const goToPreviousStep = () => {
    if (step > 1) {
      setStep(step - 1);
    }
  };

  // 예약 정보 확인 화면으로 이동 (임시)
  const viewReservation = () => {
    window.alert('예약 정보 확인 화면으로 이동합니다.');
  };

  // 이벤트 정보
  const eventInfo = {
    title: '실종',
    venue: '홍익대학교 학생회관 3층 소극장',
    period: '2025.05.14 ~ 2025.05.18',
    posterUrl: ''
  };

  // 결제 정보 계산
  const calculatePayment = () => {
    const basePrice = 10000;
    const discountAmount = discountType === 'standard' ? 3000 : 0;
    const deliveryFee = 0; // 현장수령은 0원
    
    return {
      basePrice,
      discountAmount,
      deliveryFee,
      totalPrice: basePrice - discountAmount + deliveryFee
    };
  };

  return {
    // 상태
    dateTimeOptions,
    step,
    dateTime,
    people,
    discountType,
    paymentType,
    deliveryType,
    nextActive,
    eventInfo,
    studentId,
    depositorName,
    termsAgreed,
    isPC,
    
    // 액션
    setDateTime,
    setPeople,
    setDiscountType,
    setPaymentType,
    setDeliveryType,
    goToNextStep,
    goToPreviousStep,
    viewReservation,
    setStudentId,
    setDepositorName,
    setTermsAgreed,
    
    // 유틸리티
    calculatePayment,
    getCurrentStepContent,
    getMaxSteps,
  };
};

export default useTicketing;
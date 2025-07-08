
import React from 'react';
import ActionButton from '../components/ActionButton';
import { SuccessMessage } from '../styles/commonStyles';
import useResponsive from '../hooks/useResponsive';

const Step3 = ({ ticketing: { viewReservation } }) => {
  const isPC = useResponsive();

  return (
    <SuccessMessage>
      {/* PC 버전 메시지 */}
      {isPC && (
        <p style={{
          fontSize: '24px',
          fontWeight: 'bold',
          color: '#F67676',
          marginBottom: '60px'
        }}>
          예약이 확정되었습니다!
        </p>
      )}
      {/* 모바일 버전 메시지 */}
        {!isPC && (
          <p>예매 완료!</p>
        )}
      <ActionButton isActive={true} onClick={viewReservation} className="bottom">
        예약현황 보러가기
      </ActionButton>
    </SuccessMessage>
  );
};

export default Step3;

import styled from 'styled-components';

// 반응형 미디어쿼리 상수
const media = {
  mobile: `@media (max-width: 767px)`,
  pc: `@media (min-width: 768px)`,
};

export const BookingSummary = styled.div`
  margin-top: 20px;
`;

export const SummaryRow = styled.div`
  position: relative;
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  margin-bottom: 16px;
  
  ${media.pc} {
    margin-top: 0px;
    margin-bottom: 20px;
  }
  
  &.total {
    margin-top: 16px;
    font-weight: bold;
    color: #ff6b6b;

    ${media.pc} {
      margin-top: 60px;
    }
  }
  
  div {
    font-size: 14px;
    font-weight: 500;
    color: #929292;
    display: block;
    margin-bottom: 8px;

    ${media.pc} {
      font-size: 16px;
      font-weight: bold;
      color: #000000;
    }
  }

  span {
    position: absolute;
    left: 100px;
    ${media.pc} { left: 120px; }
    &.discount {
      color: #ff6b6b;
    }
  }
`;
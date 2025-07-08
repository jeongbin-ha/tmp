
import React from 'react';
import { PosterContainer, Poster, EventTitle, EventVenue } from '../styles/commonStyles';
import PosterImg from './icons/PosterImg.svg';
import useResponsive from '../hooks/useResponsive';

const PosterInfo = ({ eventInfo }) => {
  const { title, venue, posterUrl } = eventInfo;
  const isPC = useResponsive();

  return (
    <PosterContainer>
      <Poster src={posterUrl || PosterImg} alt={`${title} 포스터`} />
      {!isPC && (
        <>
        <EventTitle>{title}</EventTitle>
        <EventVenue>{venue}</EventVenue>
        </>
      )}
    </PosterContainer>
  );
};

export default PosterInfo;

import { useState, useEffect } from 'react';

const useResponsive = () => {
  const [isPC, setIsPC] = useState(false);
  
  useEffect(() => {
    const checkDevice = () => {
      setIsPC(window.innerWidth >= 768);
    };

    checkDevice();
    window.addEventListener('resize', checkDevice);
    
    return () => window.removeEventListener('resize', checkDevice);
  }, []);

  return isPC;
};

export default useResponsive;
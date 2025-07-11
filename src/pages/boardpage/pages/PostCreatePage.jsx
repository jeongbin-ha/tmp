
import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Container } from '../styles/commonStyles';
import {
  ContentArea,
  FormContainer,
  CategoryLabel,
  FormField,
  TitleInput,
  ContentTextarea,
  ImageSection,
  ImageAddButton,
  ImageAddIcon,
  ImageAddText,
  ImagePreviewContainer,
  ImagePreview,
  ImageDeleteButton,
  RegisterBtnContainer,
  RegisterButton,
} from '../styles/formStyles';
import Header from '../components/BoardHeader';
import Modal from '../components/Modal';
import useModal from '../hooks/useModal';
import usePosts from '../hooks/usePosts';
import Camera from '../components/Icons/Camera.svg';
import useResponsive from '../hooks/useResponsive'

const PostCreatePage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { addPost } = usePosts();
  const isPC = useResponsive();

  // URL에서 카테고리 정보 가져오기 (플로팅 버튼에서 전달받음)
  const searchParams = new URLSearchParams(location.search);
  const category = searchParams.get('category') || 'general';
  
  const [formData, setFormData] = useState({
    title: '',
    content: '',
    category: category,
    images: []
  });

  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [historyPushed, setHistoryPushed] = useState(false);  //히스토리 엔트리 추가 여부를 추적
  const { isOpen: isExitModalOpen, openModal: openExitModal, closeModal: closeExitModal } = useModal();

  // hasUnsavedChanges 계산을 별도 useEffect로
  useEffect(() => {
    const hasContent = formData.title.trim() || formData.content.trim() || formData.images.length > 0;
    setHasUnsavedChanges(hasContent);
  }, [formData]);

  // 브라우저 이벤트 처리
  useEffect(() => {
    // 페이지 진입 시 히스토리 엔트리를 한 번만 추가
    if (!historyPushed) {
      window.history.pushState(null, '', window.location.href);
      setHistoryPushed(true);
    }

    // 브라우저 새로고침/탭 닫기 방지
    const handleBeforeUnload = (e) => {
      if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = '';
      }
    };

    // 브라우저 뒤로가기 방지
    const handlePopState = (e) => {
      e.preventDefault(); // 항상 뒤로가기 방지
      
      if (hasUnsavedChanges) {
        openExitModal(); // 작성 중이면 확인 모달
      } else {
        navigate('/board'); // 빈 상태면 바로 이동
      }
    };

    // 이벤트 리스너 등록
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('popstate', handlePopState);

    // 클린업 함수
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('popstate', handlePopState);
    };
  }, [hasUnsavedChanges, openExitModal, navigate, historyPushed]); // 필요한 의존성만 포함

  // 카테고리 표시 텍스트
  const getCategoryDisplayName = (cat) => {
    switch(cat) {
      case 'general': return '일반 게시판';
      case 'promotion': return '홍보 게시판';
      case 'hot': return '일반 게시판'; // Hot탭에서 온 경우도 일반으로
      default: return '일반 게시판';
    }
  };

  // 입력값 변경 핸들러
  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // 이미지 추가 핸들러
  const handleImageAdd = (event) => {
    const files = Array.from(event.target.files);
    const remainingSlots = 5 - formData.images.length;
    const filesToAdd = files.slice(0, remainingSlots);
    
    filesToAdd.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        setFormData(prev => ({
          ...prev,
          images: [...prev.images, {
            id: Date.now() + Math.random(),
            file: file,
            url: e.target.result
          }]
        }));
      };
      reader.readAsDataURL(file);
    });
  };

  // 이미지 삭제 핸들러
  const handleImageDelete = (imageId) => {
    setFormData(prev => ({
      ...prev,
      images: prev.images.filter(img => img.id !== imageId)
    }));
  };

  // 완료 버튼 활성화 조건
  const isFormValid = formData.title.trim() && formData.content.trim();

  // 게시글 작성 완료
  const handleSubmit = async () => {
    if (!isFormValid) return;

    try {
      // 새 게시글 생성
      const newPost = {
        title: formData.title.trim(),
        content: formData.content.trim(),
        category: formData.category === 'hot' ? 'general' : formData.category,
        images: formData.images.map(img => img.url), // 실제로는 서버에 업로드 후 URL을 받아야 함
        author: '익명',
        date: new Date().toLocaleDateString('ko-KR', { 
          year: 'numeric', 
          month: '2-digit', 
          day: '2-digit' 
        }).replace(/\. /g, '.').replace('.', ''),
        likes: 0,
        comments: 0,
        isHot: false,
        userId: 'currentUser' // 실제로는 로그인한 사용자 ID
      };

      const createdPost = await addPost(newPost);
      
      // 작성한 게시글 상세 페이지로 이동
      navigate(`/board/create/success?postId=${createdPost.id}`);
    } catch (error) {
      console.error('게시글 작성 실패:', error);
      // 에러 처리 (토스트 메시지 등)
    }
  };

  // 뒤로가기 핸들러
  const handleBack = () => {
    if (hasUnsavedChanges) {
      openExitModal(); // 작성 중이면 확인 모달
    } else {
      navigate('/board'); // 빈 상태면 바로 이동
    }
  };

  // 나가기 확인 모달 액션 
  const exitModalActions = [
    {
      label: '취소',
      type: 'cancel',
      onClick: closeExitModal
    },
    {
      label: '나가기',
      type: 'confirm',
      onClick: () => {
        closeExitModal();
        navigate('/board');
      }
    }
  ];

  return (
    <Container>
      {!isPC && (
        <Header
          title="create"
          showBack={true}
          onBack={handleBack}
          onComplete={handleSubmit}
          completeDisabled={!isFormValid}
        />
      )}

      <ContentArea>
        <FormContainer>
          <FormField>
            <RegisterBtnContainer>
              <TitleInput
                value={formData.title}
                onChange={(e) => handleInputChange('title', e.target.value)}
                placeholder="제목을 입력하세요"
                maxLength={100}
              />
              {isPC && ( <RegisterButton disabled={!isFormValid} onClick={isFormValid ? handleSubmit : undefined} >등록</RegisterButton> )}
            </RegisterBtnContainer>
          </FormField>

          <CategoryLabel>{getCategoryDisplayName(formData.category)}</CategoryLabel>
          {isPC && (
            <div style={{display: 'flex', justifyContent: 'end'}}>
              <ImageAddButton disabled={formData.images.length >= 5}>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleImageAdd}
                  style={{ display: 'none' }}
                  id="image-upload"
                  disabled={formData.images.length >= 5}
                />
                <label htmlFor="image-upload" style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px', 
                  cursor: formData.images.length >= 5 ? 'not-allowed' : 'pointer',
                  opacity: formData.images.length >= 5 ? 0.5 : 1
                }}>
                  <ImageAddIcon src={Camera} alt="사진 추가" />
                  <ImageAddText>사진</ImageAddText>
                </label>
              </ImageAddButton>
            </div>
          )}

          <FormField>
            <ContentTextarea
              value={formData.content}
              onChange={(e) => handleInputChange('content', e.target.value)}
              placeholder="다양한 사람들과 공연에 관해 이야기를 나눠봐요!"
              rows={8}
            />
          </FormField>

          <ImageSection>
            {!isPC && ( 
              <ImageAddButton disabled={formData.images.length >= 5}>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleImageAdd}
                  style={{ display: 'none' }}
                  id="image-upload"
                  disabled={formData.images.length >= 5}
                />
                <label htmlFor="image-upload" style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px', 
                  cursor: formData.images.length >= 5 ? 'not-allowed' : 'pointer',
                  opacity: formData.images.length >= 5 ? 0.5 : 1
                }}>
                  <ImageAddIcon src={Camera} alt="사진 추가" />
                  <ImageAddText>사진</ImageAddText>
                </label>
              </ImageAddButton>
            )}
            
            {formData.images.length > 0 && (
              <ImagePreviewContainer>
                {formData.images.map((image) => (
                  <ImagePreview key={image.id}>
                    <img src={image.url} alt="미리보기" />
                    <ImageDeleteButton onClick={() => handleImageDelete(image.id)}>
                      ✕
                    </ImageDeleteButton>
                  </ImagePreview>
                ))}
              </ImagePreviewContainer>
            )}
          </ImageSection>
        </FormContainer>
      </ContentArea>

      {/* 나가기 확인 모달 */}
      <Modal
        isOpen={isExitModalOpen}
        onClose={closeExitModal}
        title="작성을 취소하시겠어요?"
        actions={exitModalActions}
      />
    </Container>
  );
};

export default PostCreatePage;
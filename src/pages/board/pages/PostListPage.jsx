
import React, { useState, useEffect } from 'react';
import {
  Container,
  ContentArea,
  LoadingSpinner,
  EmptyState,
  LoadMoreButton,
  LoadMoreContainer,
  SideMenuWrapper 
} from '../styles/commonStyles';
import HomeIconMenu from '../../../components/HomeIconMenu';
import Header from '../components/BoardHeader';
import TabBar from '../components/TabBar';
import SearchBar from '../components/SearchBar';
import PostItem from '../components/PostItem';
import FloatingButton from '../components/FloatingButton';
import usePosts from '../hooks/usePosts';
import useInfiniteScroll from '../hooks/useInfiniteScroll';
import useResponsive from '../hooks/useResponsive'

const PostListPage = () => {
  const [activeTab, setActiveTab] = useState('general');
  const isPC = useResponsive();

  const {
    posts, loading, hasMore, userType,
    canCreatePost, loadPosts, loadMore, setPage, loadMoreForPC, hasMoreForPC
  } = usePosts();

  const { lastElementRef } = useInfiniteScroll(
    () => loadMore(activeTab),
    hasMore,
    loading,
    !isPC
  );

  useEffect(() => {
    setPage(1);
    loadPosts(activeTab, 1, true);
  }, [activeTab, loadPosts, setPage]);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  const handleSearch = (searchTerm) => {
    console.log('검색:', searchTerm);
    // 검색 기능 구현
  };

  const handleLoadMoreForPC = () => {
    loadMoreForPC();
  };

  const showFloatingButton = canCreatePost(activeTab);

  return (
    <div>
      <SideMenuWrapper>
        <HomeIconMenu isWeb={true} />
      </SideMenuWrapper>

      <Container>
        {!isPC && (
          <Header 
            title="board" 
            onMenuClick={() => console.log('메뉴 클릭')}
            onSearchClick={() => console.log('검색 클릭')}
          />
        )}
        
        <TabBar 
          activeTab={activeTab} 
          onTabChange={handleTabChange} 
          showFloatingButton={showFloatingButton}
        />
        
        <SearchBar onSearch={handleSearch} />
        
        <ContentArea>
          {posts.length === 0 && !loading ? (
            <EmptyState>
              <p>게시글이 없습니다.</p>
              <p>첫 게시글을 작성해보세요!</p>
            </EmptyState>
          ) : (
            <>
              {posts.map((post, index) => (
                <PostItem
                  key={post.id}
                  post={post}
                  isLast={index === posts.length - 1}
                  lastElementRef={!isPC ? lastElementRef : null}
                />
              ))}

              {isPC && hasMoreForPC() && !loading && (
                <LoadMoreContainer>
                  <LoadMoreButton onClick={handleLoadMoreForPC}>
                    게시글 더보기
                  </LoadMoreButton>
                </LoadMoreContainer>
              )}
              
              {loading && (
                <LoadingSpinner>
                  <p>로딩 중...</p>
                </LoadingSpinner>
              )}
            </>
          )}
        </ContentArea>
        
        {!isPC && (
          <FloatingButton 
            show={showFloatingButton} 
            category={activeTab} 
          />
        )}
      </Container>
    </div>
  );
};

export default PostListPage;
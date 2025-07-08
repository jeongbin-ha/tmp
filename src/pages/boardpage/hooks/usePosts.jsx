
import { useState, useEffect, useCallback } from 'react';
import SampleImg from '../components/Icons/SampleImg.svg'
import useResponsive from './useResponsive';

// Mock 데이터
const mockPosts = [
  {
    id: 1,
    title: '홍익극연구회 <실종>',
    content: '저희 홍익극연구회에서 새로운 연극 <실종>을 3일간 공연합니다! 관심 있으신 분들의 많은 참여 부탁드립니다.\n\n공연 일정:\n- 날짜: 2025년 4월 22일 ~ 24일\n- 시간: 오후 7시 30분\n- 장소: 홍익대학교 대학로 캠퍼스 아트센터\n\n티켓 예매는 홍익대학교 연극동아리 공식 인스타그램에서 가능합니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 36,
    comments: 3,
    category: 'promotion',
    isHot: true,
    image: [SampleImg, SampleImg, SampleImg],
    userId: 'user1'
  },
  {
    id: 2,
    title: '이거슨 익명게시판',
    content: '어제 알라딘 공연에서 아이폰 16pro를 찾았는데 혹시 주인이신분 찾지마세요, 내가 가질게요 수고',
    author: '익명',
    date: '2025.04.20',
    likes: 17,
    comments: 3,
    category: 'general',
    isHot: true,
    image: SampleImg,
    userId: 'user2'
  },
  {
    id: 3,
    title: '익게로 놀러와',
    content: '알라딘 후기 좀 알려봐. 어떤거 같어?\n\n최근에 다녀온 사람들 후기 공유해주세요! 볼만한지 궁금합니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 6,
    comments: 3,
    category: 'general',
    isHot: false,
    image: [],
    userId: 'user3'
  },
  {
    id: 4,
    title: '블루스퀘이',
    content: '블루스퀘이 어제 1층 F구역 10열에 앉았는데 진짜 시야 너무 좋고 짱구는 못말려 봉미선도 못말려',
    author: '익명',
    date: '2025.04.20',
    likes: 11,
    comments: 3,
    category: 'general',
    isHot: true,
    image: [],
    userId: 'user4'
  },
  {
    id: 5,
    title: '어거 재밌다',
    content: '알라딘 공연에서 아이폰 16pro를 두고 온거 같은데 혹시 발견하시면 댓글좀 달아주세요',
    author: '익명',
    date: '2025.04.20',
    likes: 5,
    comments: 3,
    category: 'general',
    isHot: false,
    image: [],
    userId: 'user5'
  },
  {
    id: 6,
    title: '이거 재밌다',
    content: '알라딘 공연에서 아이폰 16pro를 두고 온거 같은데 혹시 발견하시면 댓글좀 달아주세요',
    author: '익명',
    date: '2025.04.20',
    likes: 1,
    comments: 3,
    category: 'general',
    isHot: false,
    image: [],
    userId: 'user6'
  },
  {
    id: 7,
    title: '이거슨 익명게시판',
    content: '알라딘 공연에서 아이폰 16pro를 두고 온거 같은데 혹시 발견하시면 댓글좀 달아주세요',
    author: '익명',
    date: '2025.04.20',
    likes: 14,
    comments: 3,
    category: 'general',
    isHot: true,
    image: [],
    userId: 'user7'
  },
  {
    id: 8,
    title: '익게로 놀러와',
    content: '알라딘 후기 좀 알려봐. 어떤거 같어?\n\n최근에 다녀온 사람들 후기 공유해주세요! 볼만한지 궁금합니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 1,
    comments: 3,
    category: 'general',
    isHot: false,
    image: [],
    userId: 'user8'
  },
  {
    id: 9,
    title: '홍익극연구회 <실종>',
    content: '저희 홍익극연구회에서 새로운 연극 <실종>을 3일간 공연합니다! 관심 있으신 분들의 많은 참여 부탁드립니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 2,
    comments: 3,
    category: 'promotion',
    isHot: false,
    image: [],
    userId: 'organizer1'
  },
  {
    id: 10,
    title: '블루스퀘이',
    content: '블루스퀘이 어제 1층 F구역 10열에 앉았는데 진짜 시야 너무 좋고 짱구는 못말려 봉미선도 못말려',
    author: '익명',
    date: '2025.04.20',
    likes: 3,
    comments: 3,
    category: 'general',
    isHot: false,
    image: [],
    userId: 'user10'
  },
  {
    id: 11,
    title: '홍익극연구회 <실종>',
    content: '저희 홍익극연구회에서 새로운 연극 <실종>을 3일간 공연합니다! 관심 있으신 분들의 많은 참여 부탁드립니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 4,
    comments: 3,
    category: 'promotion',
    isHot: false,
    image: [],
    userId: 'organizer1'
  },
  {
    id: 12,
    title: '홍익극연구회 <실종>',
    content: '저희 홍익극연구회에서 새로운 연극 <실종>을 3일간 공연합니다! 관심 있으신 분들의 많은 참여 부탁드립니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 4,
    comments: 3,
    category: 'promotion',
    isHot: false,
    image: [],
    userId: 'organizer1'
  },
  {
    id: 13,
    title: '홍익극연구회 <실종>',
    content: '저희 홍익극연구회에서 새로운 연극 <실종>을 3일간 공연합니다! 관심 있으신 분들의 많은 참여 부탁드립니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 3,
    comments: 3,
    category: 'promotion',
    isHot: false,
    image: [],
    userId: 'organizer1'
  },
  {
    id: 14,
    title: '홍익국연구회 <실종>',
    content: '저희 홍익극연구회에서 새로운 연극 <실종>을 3일간 공연합니다! 관심 있으신 분들의 많은 참여 부탁드립니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 2,
    comments: 3,
    category: 'promotion',
    isHot: false,
    image: [],
    userId: 'organizer1'
  },
  {
    id: 15,
    title: '홍익극연구회 <실종>',
    content: '저희 홍익극연구회에서 새로운 연극 <실종>을 3일간 공연합니다! 관심 있으신 분들의 많은 참여 부탁드립니다.',
    author: '익명',
    date: '2025.04.20',
    likes: 1,
    comments: 3,
    category: 'promotion',
    isHot: false,
    image: [],
    userId: 'organizer1'
  }
];

const mockComments = {
  1: [
    {
      id: 1,
      author: '익명',
      content: '와 기대된다!',
      date: '04.22',
      userId: 'user11',
      replyLevel: 0,
      parentId: null,
      likes: 8
    },
    {
      id: 2,
      author: '카모이',
      content: '꼭 보러 갈게요',
      date: '04.22',
      userId: 'user12',
      replyLevel: 0,
      parentId: null,
      likes: 1
    },
    {
      id: 3,
      author: '기돼',
      content: '티켓 예매 어떻게 하나요?',
      date: '04.22',
      userId: 'user13',
      replyLevel: 0,
      parentId: null,
      likes: 3
    }
  ],
  2: [
    {
      id: 4,
      author: '익명',
      content: '저도 궁금해요!',
      date: '04.22',
      userId: 'user14',
      replyLevel: 0,
      parentId: null,
      likes: 2
    }
  ],
  4: [
    {
      id: 5,
      author: '익명',
      content: '저도 거기 갔었는데 정말 좋더라구요',
      date: '04.22',
      userId: 'user15',
      replyLevel: 0,
      parentId: null,
      likes: 11
    },
    {
      id: 6,
      author: '익명',
      content: '다음에 또 가고 싶어요',
      date: '04.22',
      userId: 'user16',
      replyLevel: 0,
      parentId: null,
      likes: 1
    },
    {
      id: 7,
      author: '익명',
      content: 'ㅋㅋㅋ',
      date: '04.22',
      userId: 'user17',
      replyLevel: 0,
      parentId: null,
      likes: 0
    }
  ]
};

const usePosts = () => {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(1);
  const [userType] = useState('registerUser'); // 'normalUser' | 'registerUser'   - 임시, 추후 api
  const [displayedPosts, setDisplayedPosts] = useState([]);   // PC용 표시될 게시글 배열
  const [allPostsLoaded, setAllPostsLoaded] = useState([]);   // PC에서 전체 로드된 게시글 저장
  const isPC = useResponsive();

  /*
  const getUserType = useCallback(async () => {
    try {
      // const response = await api.get('/user/profile');
      // return response.data.userType;
      return 'normalUser'; // 임시
    } catch (error) {
      console.error('Failed to get user type:', error);
      return 'normalUser';
    }
  }, []);
  */
 const canCreatePost = useCallback((category) => {
    if (category === 'promotion') {
      return userType === 'registerUser';
    }
    return true; // 일반탭, Hot탭은 모든 유저 가능
  }, [userType]);

  const loadPosts = useCallback(async (category = 'general', pageNum = 1, reset = false) => {
    setLoading(true);
    
    // 시뮬레이트된 API 호출
    await new Promise(resolve => setTimeout(resolve, 100));
    
    let filteredPosts = mockPosts;
    
    if (category === 'hot') {
      filteredPosts = mockPosts.filter(post => post.isHot);
    } else if (category !== 'general') {
      filteredPosts = mockPosts.filter(post => post.category === category);
    } else {
      filteredPosts = mockPosts.filter(post => post.category === 'general');
    }
    
    const itemsPerPage = 6;
    const startIndex = (pageNum - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const newPosts = filteredPosts.slice(startIndex, endIndex);
    
    if (reset) {
      setPosts(newPosts);
      if (isPC) { // PC에서는 전체 데이터를 별도로 저장하고, 표시용 배열은 첫 6개만
        setAllPostsLoaded(filteredPosts); 
        setDisplayedPosts(newPosts);
      }
    } else {
      setPosts(prev => [...prev, ...newPosts]);
      if (isPC) {
        setDisplayedPosts(prev => [...prev, ...newPosts]);    // PC에서는 displayedPosts에 추가
      }
    }
    
    setHasMore(endIndex < filteredPosts.length);
    setLoading(false);
  }, [isPC]);

  // PC용 더보기 함수
  const loadMoreForPC = useCallback(() => {
    if (loading) return;
    
    const currentDisplayed = displayedPosts.length;
    const nextBatch = allPostsLoaded.slice(currentDisplayed, currentDisplayed + 6);
    
    if (nextBatch.length > 0) {
      setDisplayedPosts(prev => [...prev, ...nextBatch]);
    }
  }, [allPostsLoaded, displayedPosts.length, loading]);

  // PC용 더보기 가능 여부 확인
  const hasMoreForPC = useCallback(() => {
    return displayedPosts.length < allPostsLoaded.length;
  }, [displayedPosts.length, allPostsLoaded.length]);


  const loadMore = useCallback((category) => {
    if (!loading && hasMore && !isPC) {
      const nextPage = page + 1;
      setPage(nextPage);
      loadPosts(category, nextPage, false);
    }
  }, [loading, hasMore, page, loadPosts, isPC]);

  // 반응형 전환 시 상태 동기화
  useEffect(() => {
    if (isPC && posts.length > 0) {
      // PC로 전환 시: 현재 posts를 displayedPosts로 설정
      setDisplayedPosts(posts.slice(0, 6));
    }
  }, [isPC, posts]);


  const getPost = useCallback((id) => {
    return mockPosts.find(post => post.id === parseInt(id));
  }, []);

  const getComments = useCallback((postId) => {
    return mockComments[postId] || [];
  }, []);

  const updatePost = useCallback((postId, updates) => {
    const postIndex = mockPosts.findIndex(post => post.id === parseInt(postId));
    if (postIndex !== -1) {
      mockPosts[postIndex] = { ...mockPosts[postIndex], ...updates };
    }
  }, []);

  const deletePost = useCallback((postId) => {
    const index = mockPosts.findIndex(post => post.id === parseInt(postId));
    if (index !== -1) {
      mockPosts.splice(index, 1);
    }
  }, []);

  const addComment = useCallback((postId, comment) => {
    if (!mockComments[postId]) {
      mockComments[postId] = [];
    }
    mockComments[postId].push(comment);
  }, []);

  const updateComment = useCallback((postId, commentId, updates) => {
    if (mockComments[postId]) {
      const commentIndex = mockComments[postId].findIndex(comment => comment.id === commentId);
      if (commentIndex !== -1) {
        mockComments[postId][commentIndex] = { ...mockComments[postId][commentIndex], ...updates };
      }
    }
  }, []);

  const deleteComment = useCallback((postId, commentId) => {
    if (mockComments[postId]) {
      mockComments[postId] = mockComments[postId].filter(comment => comment.id !== commentId);
    }
  }, []);

  const addPost = useCallback(async (postData) => {
    // 시뮬레이트 API 호출
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const newPost = {
      id: mockPosts.length + 1,
      ...postData,
      isHot: false // 새 게시글은 기본적으로 Hot이 아님
    };
    
    // mockPosts 배열 맨 앞에 추가
    mockPosts.unshift(newPost);
    
    return newPost;
  }, []);

  return {
    posts: isPC ? displayedPosts : posts,
    loading,
    hasMore: isPC ? hasMoreForPC() : hasMore,
    userType,
    allPostsLoaded,
    displayedPosts,
    loadMoreForPC,
    hasMoreForPC,
    canCreatePost,
    loadPosts,
    loadMore,
    setPage,
    getPost,
    getComments,
    updatePost,
    deletePost,
    addComment,
    updateComment,
    deleteComment,
    addPost
  };
};

export default usePosts;
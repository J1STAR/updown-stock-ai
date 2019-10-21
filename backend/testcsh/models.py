from django.db import models
from django.contrib.auth.models import User
  # Django에서 기본 제공하는 User 모델
class Post(models.Model):
  # 모델 클래스 명은 단수로
    user = models.ForeignKey(User, on_delete=models.CASCADE)
      # 왜래키 (장고에서 기본 제공하는 User 모델과 M:1 관계)
    title = models.CharField(max_length=144)
      # 길이제한 144자
    subtitle = models.CharField(blank=True, null=True, max_length=144)
      # Application단 null OK, DB단 null OK
    content = models.TextField()
      # 기본 TextField
    created_at = models.DateTimeField(auto_now_add=True)
      # 해당 레코드 생성시 현재 시간 자동 저장
    def __str__(self):
      # 해당 모델 인스턴스를 str형으로 캐스팅 시의 리턴을 정의
        return '[{}] {}'.format(self.user.username, self.title)
          # ex) "[유동호] 포스트 제목입니다."
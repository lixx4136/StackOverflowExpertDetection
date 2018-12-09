SELECT selecteduser.Id as UserId, ifnull(ratio.RatioTC, 0) as RatioTC
from [cs520umass:sotorrent_org.SelectedUsers_ansMoreThan10] as selecteduser
left join each (
  SELECT UserId, AVG(TextNum/(TextNum+CodeNum)) as RatioTC
  FROM [cs520umass:sotorrent_org.TextCodeNumberPython]   
  GROUP BY UserId
) as ratio
on selecteduser.Id = ratio.UserId


select codejoin.PostId as PostId, ifnull(TextBlock.TextNum, 0) as TextNum, ifnull(codejoin.CodeNum,0) as CodeNum, codejoin.UserId as UserId
from (select PostId, INTEGER(count(*))as TextNum
      from [cs520umass:sotorrent_org.AnsBlocks_Python] 
      where PostBlockTypeId = 1
      group by PostId) as TextBlock
Right join each(
  select Posts.PostId as PostId, codeBlock.CodeNum as CodeNum, Posts.UserId as UserId
  from [cs520umass:sotorrent_org.AnsPosts_Python]  as Posts
  left join each
    (select PostId, INTEGER(count(*) )as CodeNum
     from [cs520umass:sotorrent_org.AnsBlocks_Python]
     where PostBlockTypeId = 2
     group by PostId) as codeBlock
  on Posts.PostId = codeBlock.PostId) as codejoin
on TextBlock.PostId = codejoin.PostId
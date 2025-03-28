#!/usr/bin/env bash
## User Node Import
LOAD CSV FROM
'file:///dm_user_profile_3000.csv' AS line
CREATE (:SuperfansUser {uid: toInteger(line[0]), nickname: line[1], device_model:line[2], device_system:line[3], follow_stars_list:split(line[4], ","), publish_posts_num: toInteger(line[5]), like_posts_num: toInteger(line[6]), comment_posts_num: toInteger(line[7]),forward_posts_num: toInteger(line[8]), report_posts_num: toInteger(line[9]), last_signin_time: toInteger(line[10])})

# Post Node Import
LOAD CSV FROM
'file:///dm_dynamic_profile_10_3000.csv' AS line
CREATE (:SuperfansPost {pid: toInteger(line[0]), publish_time: toInteger(line[1]), related_stars_list:split(line[2],","), liked_num:toInteger(line[3]), commented_num:toInteger(line[4]), forwarded_num: toInteger(line[5]), iv_url: line[6], text_info: line[7]})
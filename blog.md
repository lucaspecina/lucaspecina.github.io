---
layout: default
title: "Blog"
permalink: /blog/
---

# Blog

<ul class="post-list">
{%- for post in site.posts -%}
  <li>
    <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <span class="meta">{{ post.date | date: "%b %d, %Y" }}</span>
  </li>
{%- endfor -%}
</ul>
